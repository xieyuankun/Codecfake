import raw_dataset as dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor,Wav2Vec2Config
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)

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

# prprocess asvspoof2019LA

for part_ in ["train", "dev"]:
    codecspoof_raw = dataset.codecfake("/data2/codecfake/", "/data2/codecfake/label/", part=part_)
    target_dir = os.path.join("/data2/xyk/codecfake/preprocess_xls-r-5", part_,
                              "xls-r-5")
    config = Wav2Vec2Config.from_json_file("/data3/xyk/huggingface/wav2vec2-xls-r-300m/config.json")                          
    processor = Wav2Vec2FeatureExtractor.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/")
    model = Wav2Vec2Model.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/").cuda()
    #processor =  Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model.config.output_hidden_states = True

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for idx in tqdm(range(len(codecspoof_raw))):
        waveform, filename, label = codecspoof_raw[idx]
        waveform = pad_dataset(waveform)
        
        input_values = processor(waveform, sampling_rate=16000,
                                    return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            wav2vec2 = model(input_values).hidden_states[5].cpu()
        print(wav2vec2.shape)
        print(wav2vec2.device)
        torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (idx, filename, label)))

    print("Done")


# for part_ in ["train", "dev"]:
#     asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data2/xyk/asv2019/LA",
#                                            "/data2/xyk/asv2019/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
#     target_dir = os.path.join("/data2/xyk/asv2019/preprocess_xls-r-5", part_,
#                               "xls-r-5")
#     processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
#     model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
#     model.config.output_hidden_states = True
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     for idx in tqdm(range(len(asvspoof_raw))):
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = pad_dataset(waveform, 64600)
#         input_values = processor(waveform, sampling_rate=16000,
#                                  return_tensors="pt").input_values.cuda()
#
#         with torch.no_grad():
#             wav2vec2 = model(input_values).hidden_states[5].cpu()
#         print(wav2vec2.shape)
#         torch.save(wav2vec2, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")
#
# for part_ in ["train", "dev"]:
#     asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data2/xyk/asv2019/LA",
#                                            "/data2/xyk/asv2019/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
#     target_dir = os.path.join("/data2/xyk/asv2019/preprocess_mel", part_,
#                               "mel")
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     for idx in tqdm(range(len(asvspoof_raw))):
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         waveform = pad_dataset(waveform)
#         print(waveform.shape)
#         mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=512, hop_length=128, sample_rate=16000)(waveform)
#         mel = mel.unsqueeze(dim=0)
#         print(mel.shape)
#         torch.save(mel, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")