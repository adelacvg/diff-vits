from glob import glob
import json
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
import torchaudio.transforms as T
from text import cleaned_text_to_sequence, get_bert
import commons


class TextAudioDataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, cfg):
        self.audiopaths = glob(os.path.join(cfg['data']['training_files'], "**/*.wav"), recursive=True)\
        + glob(os.path.join(cfg['data']['training_files'], "*.wav"), recursive=True)
        self.sampling_rate = cfg['data']['sampling_rate']
        self.hop_length = cfg['data']['hop_length']
        self.add_blank = cfg['data']['add_blank']
        self.min_text_len = cfg['data']['min_text_len']
        self.max_text_len = cfg['data']['max_text_len']
    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio
        spec_filename = filename.replace(".wav", ".spec.pt")
        mel_filename = filename.replace(".wav", ".mel.pt")
        spec = torch.load(spec_filename)
        mel = torch.load(mel_filename)
        return spec, mel, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        word2ph = [i for i in word2ph]
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

        if self.add_blank:
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
    def get_audio_text_pair(self, audiopath, split='|'):
        text_path = audiopath.replace('.wav','.text')
        with open(text_path, encoding='utf-8') as f:
            texts = f.readline().strip().split(split)
        # separate filename, speaker_id and text
        language, text, phones, tone, word2ph = texts
        phones = phones.split(" ")
        tone = [int(i) for i in tone.split(" ")]
        word2ph = [int(i) for i in word2ph.split(" ")]

        phones, tone, language = self.get_text(text, word2ph, phones, tone, language, audiopath)

        spec, mel, wav = self.get_audio(audiopath)
        return (phones, spec, mel, wav, tone, language)
    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths[index])

    def __len__(self):
        return len(self.audiopaths)

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_mel_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_mel_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        mel_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            mel = row[2]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            tone = row[4]
            tone_padded[i, :tone.size(0)] = tone

            language = row[5]
            language_padded[i, :language.size(0)] = language

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, tone_padded, language_padded
cfg = json.load(open('./config.json'))
train_dataset = TextAudioDataset(cfg)
phones, spec, mel, wav, tone, language = train_dataset[0]
