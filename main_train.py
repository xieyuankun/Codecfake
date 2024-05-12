import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from CSAM import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler

from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import eval_metrics as em

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/data2/xyk/codecfake')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/data2/xyk/codecfake/preprocess_xls-r-5')
    
    parser.add_argument("-f1", "--path_to_features1", type=str, help="cotrain_dataset1_path",
                        default='/data2/xyk/asv2019/preprocess_xls-r-5')
    
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r-5',
                        choices=["mel", "xls-r-5"])
    parser.add_argument("--feat_len", type=int, help="features length", default=201)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=False,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")

    parser.add_argument('-m', '--model', help='Model arch', default='W2VAASIST',
                        choices=['lcnn','W2VAASIST'])

    # Training hyperparameters
    parser.add_argument('--train_task', type=str, default='co-train', choices=['19LA','codecfake','co-train'], help="training dataset")

    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=2, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")

    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    # generalized strategy 
    parser.add_argument('--SAM', type= bool, default= False, help="use SAM")
    parser.add_argument('--ASAM', type= bool, default= False, help="use ASAM")
    parser.add_argument('--CSAM', type= bool, default= False, help="use CSAM")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat,  labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, labels


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'W2VAASIST':
        feat_model = W2VAASIST().cuda()

    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    #feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    
    if args.SAM or args.CSAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    if args.ASAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            adaptive = True,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    if args.train_task == '19LA':
        asv_training_set = ASVspoof2019(args.access_type, args.path_to_features1, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        asv_validation_set = ASVspoof2019(args.access_type, args.path_to_features1, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        trainOriDataLoader = DataLoader(asv_training_set, batch_size=int(args.batch_size),
                                        shuffle=False, num_workers=args.num_workers,
                                        sampler=torch_sampler.SubsetRandomSampler(range(25380)))
        valOriDataLoader = DataLoader(asv_validation_set, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=args.num_workers,
                                      sampler=torch_sampler.SubsetRandomSampler(range(24844)))

    if args.train_task == 'codecfake':
        codec_training_set = codecfake(args.access_type, args.path_to_features, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        codec_validation_set = codecfake(args.access_type, args.path_to_features, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        trainOriDataLoader = DataLoader(codec_training_set, batch_size=int(args.batch_size ),
                                        shuffle=False, num_workers=args.num_workers, persistent_workers=True,pin_memory= True,
                                        sampler=torch_sampler.SubsetRandomSampler(range(740747)))
        valOriDataLoader = DataLoader(codec_validation_set, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=args.num_workers,persistent_workers=True,
                                      sampler=torch_sampler.SubsetRandomSampler(range(92596)))

    if args.train_task == 'co-train':
        # domain_19train,dev
        asv_training_set = ASVspoof2019(args.access_type, args.path_to_features1, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        asv_validation_set = ASVspoof2019(args.access_type, args.path_to_features1, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

        # domain_codectrain, dev
        codec_training_set = codecfake(args.access_type, args.path_to_features, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        codec_validation_set = codecfake(args.access_type, args.path_to_features, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

        # concat dataset
        training_set = ConcatDataset([codec_training_set, asv_training_set])
        validation_set = ConcatDataset([codec_validation_set, asv_validation_set])

        train_total_samples_codec = len(codec_training_set)
        train_total_samples_asv = len(asv_training_set)
        train_total_samples_combined = len(training_set)
        train_codec_weight = train_total_samples_codec / train_total_samples_combined
        train_asv_weight = train_total_samples_asv / train_total_samples_combined

        if args.CSAM:
            trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                                            shuffle=False, num_workers=args.num_workers,
                                            sampler=CSAMSampler(dataset=training_set,
                                batch_size=int(args.batch_size),ratio_dataset1= train_codec_weight,ratio_dataset2 = train_asv_weight))

        if args.SAM or args.ASAM:
            trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size * args.ratio),
                                shuffle=False, num_workers=args.num_workers,pin_memory=True,
                                sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))))
        valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=args.num_workers,
                                      sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))))






    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)


    weight = torch.FloatTensor([10,1]).to(args.device)   # concentrate on real 0

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight)

    else:
        criterion = nn.functional.binary_cross_entropy()

    prev_loss = 1e8

    monitor_loss = 'base_loss'

    for epoch_num in tqdm(range(args.num_epochs)):

        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                featOri, audio_fnOri,  labelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                featOri, audio_fnOri,  labelsOri = next(trainOri_flow)


            feat = featOri
            labels = labelsOri

            feat = feat.transpose(2, 3).to(args.device)
            labels = labels.to(args.device)

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()
                feat_optimizer.second_step(zero_grad=True)
            
            else:
                feat_optimizer.zero_grad()
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.backward()
                feat_optimizer.step()


            trainlossDict['base_loss'].append(feat_loss.item())

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                            str(trainlossDict[monitor_loss][-1]) + "\n")

        feat_model.eval()
        with torch.no_grad():
            ip1_loader,  idx_loader, score_loader = [],  [], []
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    featOri, audio_fnOri, labelsOri= next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    featOri, audio_fnOri, labelsOri= next(valOri_flow)
                feat = featOri
                labels = labelsOri

                feat = feat.transpose(2, 3).to(args.device)
                labels = labels.to(args.device)
                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                ip1_loader.append(feats)
                idx_loader.append((labels))


                devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)

                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '

                print(desc_str)
                scores = torch.cat(score_loader, 0).data.cpu().numpy()
                labels = torch.cat(idx_loader, 0).data.cpu().numpy()

                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" +
                                str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                                "\n")

        valLoss = np.nanmean(devlossDict[monitor_loss])
        if (epoch_num + 1) % 1 == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))

        if valLoss < prev_loss:
            # Save the model checkpoint
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            prev_loss = valLoss


    return feat_model


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)
