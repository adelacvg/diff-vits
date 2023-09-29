import json
import re
import argparse
from string import punctuation
from vocos import Vocos
import commons

import torch
import torchaudio
import torchaudio.transforms as T
import yaml
import numpy as np
import os
from torch.utils.data import DataLoader
from g2p_en import G2p
from model3 import NaturalSpeech2
from pypinyin import pinyin, Style

from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text

def preprocess(text, preprocess_config):
    text = text.strip()
    language="ZH"
    add_blank = True
    norm_text, phones, tones, word2ph = clean_text(text, language)

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phones, tones, language)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return phone, tone, language


def synthesize(model, cfg, vocos, batchs, control_values, device):
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        phoneme, tone, language, refer_path, phoneme_length = batch 
        phoneme =phoneme.to(device)
        tone = tone.to(device)
        language = language.to(device)
        phoneme_length = torch.LongTensor(phoneme_length).to(device)
        refer_audio,sr = torchaudio.load(refer_path)
        refer_audio24k = T.Resample(sr, 24000)(refer_audio)
        spec_process = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            center=True,
            power=1,
        )
        spec = spec_process(refer_audio24k).to(device)# 1 100 T
        spec = torch.log(torch.clip(spec, min=1e-7))
        refer = spec
        refer_length = torch.tensor([refer.size(1)]).to(device)
        # print(refer.shape)
        with torch.no_grad():
            samples, mel = model.sample(phoneme, refer, phoneme_length, refer_length,\
                                tone, language, vocos)
            samples = samples.detach().cpu()
    return samples
def load_model(model_path, device, cfg):
    data = torch.load(model_path, map_location=device)
    model = NaturalSpeech2(cfg=cfg)
    model.load_state_dict(data['model'])

    model.to(device)
    return model.eval()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="你好，再见。",
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="zh",
        help="language of the input text",
    )
    parser.add_argument(
        "--refer",
        type=str,
        default="138.wav",
        help="reference audio path for single-sentence mode only",
    )
    parser.add_argument(
        # "-c", "--config_path", type=str, default="config.json", help="path to config.json"
        "-c", "--config_path", type=str, default="config.json", help="path to config.json"
    )
    parser.add_argument(
        # "-m", "--model_path", type=str, default="logs/tts/model-1000.pt", help="path to model.pt"
        "-m", "--model_path", type=str, default="logs/tts/2023-09-28-19-43-44/model-172.pt", help="path to model.pt"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2",
        help="specify the device, cpu or cuda",
    )
    args = parser.parse_args()

    device = args.device
    # Check source texts
    assert args.text is not None

    # Read Config

    cfg = json.load(open(args.config_path))

    # Get model
    model = load_model(args.model_path, device, cfg)

    # Load vocoder
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    ids = raw_texts = [args.text[:100]]
    phone, tone, language = preprocess(args.text, cfg)
    phone, tone, language = phone.unsqueeze(0), tone.unsqueeze(0), language.unsqueeze(0)
    text_lens = np.array([len(phone[0])])
    raw_path = 'raw'
    refer_name = args.refer
    refer_path = f"{raw_path}/{refer_name}"
    batchs = [( phone, tone, language,refer_path,text_lens)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    audios = synthesize(model, cfg, vocos, batchs, control_values, device)

    results_folder = "output"
    result_path = f'./{results_folder}/tts_{refer_name}.wav'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    torchaudio.save(result_path, audios, 24000)
