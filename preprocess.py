import os
import argparse
from random import shuffle
import torchaudio
import torchaudio.transforms as T
import torch
from glob import glob
from tqdm import tqdm
import utils
from text.cleaner import clean_text
import numpy as np

in_dir = ""

# with open( transcription_path+'.cleaned', 'w', encoding='utf-8') as f:
#     for line in tqdm.tqdm(open(transcription_path, encoding='utf-8').readlines()):

def process_one(filename,language):
    text_path = filename.replace('.wav','.txt')
    out_text_path = text_path.replace(in_dir,in_dir.rstrip('/')+"_processed")
    if not os.path.exists(os.path.dirname(out_text_path)):
        os.makedirs(os.path.dirname(out_text_path))
    with open(out_text_path, 'w', encoding='utf-8') as f:
        line = open(text_path,encoding='utf-8').readline()
        try:
            text = line.strip()
            #language = "ZH"
            norm_text, phones, tones, word2ph = clean_text(text, language)
            f.write('{}|{}|{}|{}|{}\n'.format(language, norm_text, ' '.join(phones),
                                                    " ".join([str(i) for i in tones]),
                                                    " ".join([str(i) for i in word2ph])))
        except Exception as error :
            print("err!", filename, error)
    wav, sr = torchaudio.load(filename)
    if wav.shape[0] > 1:  # mix to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav24k = T.Resample(sr, 24000)(wav)
    filename = filename.replace(in_dir, in_dir.rstrip('/')+"_processed")
    wav24k_path = filename
    if not os.path.exists(os.path.dirname(wav24k_path)):
        os.makedirs(os.path.dirname(wav24k_path))
    torchaudio.save(wav24k_path, wav24k, 24000)

    melspec_path = filename.replace(".wav", ".mel.pt")
    melspec_process = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        center=True,
        power=1,
    )
    spec = melspec_process(wav24k)# 1 100 T
    spec = torch.log(torch.clip(spec, min=1e-7))
    torch.save(spec, melspec_path)

    spec_path = filename.replace(".wav", ".spec.pt")
    spec_process = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=256,
        center=True,
        power=1,
    )
    spec = spec_process(wav24k)# 1 100 T
    spec = torch.log(torch.clip(spec, min=1e-7))
    torch.save(spec, spec_path)


def process_batch(filenames,language):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename,language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset", help="path to input dir"
    )
    parser.add_argument(
        "--language", type=str, default="ZH", help="path to input dir"
    )
    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/**/*.wav", recursive=True)  # [:10]
    in_dir = args.in_dir
    shuffle(filenames)
    process_batch(filenames,args.language)
