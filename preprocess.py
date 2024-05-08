import raw_dataset as dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor,Wav2Vec2Config

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
    
def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data
    

for part_ in ["train", "dev"]:
    asvspoof_raw = dataset.codecfake("/data2/codecfake/codec_final/",
                                   "/data2/codecfake/codec_final/label/", part=part_)
    target_dir = os.path.join("/data2/xyk/codecfake/preprocess_xls-r-5-cpu", part_,
                              "xls-r-5")
    config = Wav2Vec2Config.from_json_file("/data3/xyk/huggingface/wav2vec2-xls-r-300m/config.json")                          
    processor = Wav2Vec2FeatureExtractor.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/")
    model = Wav2Vec2Model.from_pretrained("/data3/xyk/huggingface/wav2vec2-xls-r-300m/").cuda()
    #processor =  Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model.config.output_hidden_states = True

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for idx in tqdm(range(len(asvspoof_raw))):
        waveform, filename, label = asvspoof_raw[idx]
        waveform = pad_dataset(waveform)
        
        input_values = processor(waveform, sampling_rate=16000,
                                    return_tensors="pt").input_values.cuda()  # torch.Size([1, 31129])
        with torch.no_grad():
            wav2vec2 = model(input_values).hidden_states[5].to('cpu')
        print(wav2vec2.shape)
        print(wav2vec2.device)
        torch.save(wav2vec2.float(), os.path.join(target_dir, "%06d_%s_%s.pt" % (idx, filename, label)))

    print("Done")

