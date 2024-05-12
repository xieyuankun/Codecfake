from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import raw_dataset as dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor,Wav2Vec2Config
import numpy as np


def init():
    parser = argparse.ArgumentParser("generate model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would liek to score on",
                        required=True, default='19eval', choices=["19eval","ITW","codecfake"])
    parser.add_argument("--gpu", type=str, help="GPU index", default="2")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")


    return args


def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform


def generate_score(task, feat_model_path):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ADD_model = torch.load(feat_model_path)
    config = Wav2Vec2Config.from_json_file("/data3/xyk/huggingface/wav2vec2-xls-r-300m/config.json")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/")
    model = Wav2Vec2Model.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/").cuda()
    model.config.output_hidden_states = True
    ADD_model.eval()
    if task == '19eval':
        with open('./result/19LA_result.txt', 'w') as cm_score_file:
            asvspoof_raw = dataset.ASVspoof2019LAeval()
            for idx in tqdm(range(len(asvspoof_raw))):
                waveform, filename, labels  = asvspoof_raw[idx]
                waveform = waveform.to(device)
                waveform = pad_dataset(waveform).to('cpu')
                input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  
                with torch.no_grad():
                    wav2vec2 = model(input_values).hidden_states[5].cuda()  
                w2v2, audio_fn= wav2vec2, filename
                this_feat_len = w2v2.shape[1]
                w2v2 = w2v2.unsqueeze(dim=0)
                w2v2 = w2v2.transpose(2, 3).to(device)
                feats, w2v2_outputs = ADD_model(w2v2)
                score = F.softmax(w2v2_outputs)[:, 0]
                cm_score_file.write('%s %s %s\n' % (
                audio_fn, score.item(), "spoof" if labels== "spoof" else "bonafide"))

    if task == 'ITW':
        with open('./result/ITW_result.txt', 'w') as cm_score_file:
            ITW_raw = dataset.ITW()
            for idx in tqdm(range(len(ITW_raw))):
                waveform, filename, labels  = ITW_raw[idx]
                waveform = waveform.to(device)
                waveform = pad_dataset(waveform).to('cpu')
                input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()  
                with torch.no_grad():
                    wav2vec2 = model(input_values).hidden_states[5].cuda()  
                w2v2, audio_fn= wav2vec2, filename
                this_feat_len = w2v2.shape[1]
                w2v2 = w2v2.unsqueeze(dim=0)
                w2v2 = w2v2.transpose(2, 3).to(device)
                feats, w2v2_outputs = ADD_model(w2v2)
                score = F.softmax(w2v2_outputs)[:, 0]
                cm_score_file.write('%s %s %s\n' % (
                audio_fn, score.item(), "spoof" if labels== "spoof" else "bonafide"))

    if task == 'codecfake':
        for condition in ['C1','C2','C3','C4','C5','C6','C7','A1','A2','A3']:
            file_path = './result/{}_result.txt'.format(condition)
            with open(file_path, 'w') as cm_score_file:
                codecfake_raw = dataset.codecfake_eval(type=condition)
                for idx in tqdm(range(len(codecfake_raw))):
                    waveform, filename, labels  = codecfake_raw[idx]
                    waveform = waveform.to(device)
                    waveform = pad_dataset(waveform).to('cpu')
                    input_values = processor(waveform, sampling_rate=16000,
                                            return_tensors="pt").input_values.cuda()  
                    with torch.no_grad():
                        wav2vec2 = model(input_values).hidden_states[5].cuda()  
                    w2v2, audio_fn= wav2vec2, filename
                    this_feat_len = w2v2.shape[1]
                    w2v2 = w2v2.unsqueeze(dim=0)
                    w2v2 = w2v2.transpose(2, 3).to(device)
                    feats, w2v2_outputs = ADD_model(w2v2)
                    score = F.softmax(w2v2_outputs)[:, 0]
                    cm_score_file.write('%s %s %s\n' % (
                    audio_fn, score.item(), "fake" if labels== "fake" else "real"))


if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    generate_score(args.task, model_path)