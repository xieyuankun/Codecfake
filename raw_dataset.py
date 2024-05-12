#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
from typing import Tuple
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor,Wav2Vec2Config

torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath,sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


class ASVspoof2019Raw(Dataset):
    def __init__(self, access_type, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019Raw, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.path_to_protocol = path_to_protocol
        if self.part =='train':
            protocol = os.path.join(os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trn.txt'))
        else:
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        # if self.part == "eval":
        #     protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
        #                             '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        # # would not work if change data split but this csv is only for feat_len
        # self.csv = pd.read_csv(self.ptf + "Set_csv.csv")

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + ".flac")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)

class codecfake(Dataset):
    def __init__(self, path_to_database, path_to_protocol, part='train'):
        super(codecfake, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, self.part)
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(os.path.join(self.path_to_protocol, self.part + '.txt'))
        self.label = {"fake": 0, "real": 1}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)


class codecfake_eval(Dataset):
    def __init__(self,type):
        super(codecfake_eval, self).__init__()
        self.type = type

        self.path_to_audio = os.path.join('/data2/codecfake/codec_final',type)
        self.path_to_protocol = os.path.join('/data2/codecfake/codec_final/label',type) +'.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        filename,label,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)
    
    
class codecfake_eval_trace(Dataset):
    def __init__(self,type):
        super(codecfake_eval_trace, self).__init__()
        self.type = type
        self.path_to_audio = os.path.join('/data2/codecfake/codec_final',type)
        self.path_to_protocol = os.path.join('/data2/codecfake/codec_final/label',type) +'.txt'
        if self.type == 'C0':
            self.path_to_audio = os.path.join('/data2/codecfake/codec_final','C1')
            self.path_to_protocol = os.path.join('/data2/codecfake/codec_final/label',type) +'.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        filename,label,labeltype = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label,labeltype

    def collate_fn(self, samples):
        return default_collate(samples)
    

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


    
class LLM_eval(Dataset):
    def __init__(self,type):
        super(LLM_eval, self).__init__()
        self.type = type
        self.path_to_audio = os.path.join('/data2/codecfake/ALM',type)
        self.path_to_protocol = os.path.join('/data2/codecfake/ALM/label',type) +'.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        filename,label,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)


class LLM_eval_trace(Dataset):
    def __init__(self,type):
        super(LLM_eval_trace, self).__init__()
        self.type = type
        self.path_to_audio = os.path.join('/data2/codecfake/ALM',type)
        self.path_to_protocol = os.path.join('/data2/codecfake/ALM/label',type) +'.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        filename,label,labeltype = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label,labeltype

    def collate_fn(self, samples):
        return default_collate(samples)











class codecfake_dev(Dataset):
    def __init__(self):
        super(codecfake_dev, self).__init__()
        self.path_to_audio = '/data2/codecfake/codec_final/dev'
        self.path_to_protocol = '/data2/codecfake/codec_final/label/dev.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        filename,label,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)


















class ASVspoof2021LA(Dataset):
    def __init__(self):
        super(ASVspoof2021LA, self).__init__()
        self.path_to_audio = '/data2/xyk/asv2021/ASVspoof2021_LA_eval/flac'
        self.path_to_protocol = '/data2/xyk/asv2021/ASVspoof2021_LA_eval/trial_metadata.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        _,filename, _, _,_,label,_,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename+'.flac')
        print(filepath)
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)
        
class ASVspoof2021DF(Dataset):
    def __init__(self):
        super(ASVspoof2021DF, self).__init__()
        self.path_to_audio = '/data2/xyk/asv2021/ASVspoof2021_DF_eval/flac'
        self.path_to_protocol = '/data2/xyk/asv2021/ASVspoof2021_DF_eval/trial_metadata.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        _,filename, _, _,_,label,_,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename+'.flac')
        print(filepath)
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)
   
class ASVspoof2019LAeval(Dataset):
    def __init__(self):
        super(ASVspoof2019LAeval, self).__init__()
        self.path_to_audio = '/data2/xyk/asv2019/LA/ASVspoof2019_LA_eval/flac'
        self.path_to_protocol = '/data2/xyk/asv2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        _,filename, _, _,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename+'.flac')
        print(filepath)
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)
        
        
class ITW(Dataset):
    def __init__(self):
        super(ITW, self).__init__()
        self.path_to_audio = '/data2/xyk/ITW/wav'
        self.path_to_protocol = '/data2/xyk/ITW/label.txt'
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        print(self.all_info[idx],"self.all_info[idx]")
        _,filename, _, _,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename+'.wav')
        print(filepath)
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)
                                

