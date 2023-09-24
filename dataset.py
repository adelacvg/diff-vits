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
from torch.utils.data import Dataset, DataLoader


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
        spec = torch.load(spec_filename).squeeze(0)
        mel = torch.load(mel_filename).squeeze(0)
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
        text_path = audiopath.replace('.wav','.txt')
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

class TextAudioCollate():
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

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, mel_padded, tone_padded, language_padded

class TextAudioDataset_split(torch.utils.data.Dataset):
    def __init__(self, cfg, all_in_mem: bool = False):
        self.audiopaths = glob(os.path.join(cfg['data']['training_files'], "**/*.wav"), recursive=True)\
        + glob(os.path.join(cfg['data']['training_files'], "*.wav"), recursive=True)
        self.sampling_rate = cfg['data']['sampling_rate']
        self.hop_length = cfg['data']['hop_length']
        self.add_blank = cfg['data']['add_blank']
        self.min_text_len = cfg['data']['min_text_len']
        self.max_text_len = cfg['data']['max_text_len']
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        audio = T.Resample(sampling_rate, self.sampling_rate)(audio)

        mel = torch.load(filename.replace(".wav", ".mel.pt")).squeeze(0)

        return mel.detach(), audio.detach()
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
        text_path = audiopath.replace('.wav','.txt')
        with open(text_path, encoding='utf-8') as f:
            texts = f.readline().strip().split(split)
        # separate filename, speaker_id and text
        language, text, phones, tone, word2ph = texts
        phones = phones.split(" ")
        tone = [int(i) for i in tone.split(" ")]
        word2ph = [int(i) for i in word2ph.split(" ")]

        phones, tone, language = self.get_text(text, word2ph, phones, tone, language, audiopath)

        mel, wav = self.get_audio(audiopath)
        return phones, mel, wav, tone, language

    def random_slice(self, phone, mel, audio, tone, language):
        if mel.shape[1] < 30:
            print("skip too short audio")
            return None
        if mel.shape[1] > 400:
            start = random.randint(0, mel.shape[1]-400)
            end = start + 400
            mel= mel[:, start:end]
            audio = audio[:, start * self.hop_length : end * self.hop_length]
        len_mel = mel.shape[1]
        l = random.randint(int(len_mel//3), int(len_mel//3*2))
        u = random.randint(0, len_mel-l)
        v = u + l
        refer1 = mel[:, u:v]
        refer2 = torch.cat([mel[:, :u], mel[:, v:]], dim=-1)
        spec = mel
        # audio = torch.cat([audio[:, :u * self.hop_length], audio[:, v * self.hop_length:]], dim=-1)
        audio = audio
        return phone, spec, refer1, refer2, audio, tone, language

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio_text_pair(self.audiopaths[index]))
        # print(1)

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate_split():

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].shape[1] for x in batch]),
            dim=0, descending=True)

        # phone, spec, refer, wav, tone, language 
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_refer1_len = max([x[2].size(1) for x in batch])
        max_refer2_len = max([x[3].size(1) for x in batch])
        max_wav_len = max([x[4].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        refer1_lengths = torch.LongTensor(len(batch))
        refer2_lengths = torch.LongTensor(len(batch))

        spec_dim = batch[0][1].shape[0]
        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), spec_dim, max_spec_len+1)
        refer1_padded = torch.FloatTensor(len(batch), spec_dim, max_refer1_len+1)
        refer2_padded = torch.FloatTensor(len(batch), spec_dim, max_refer2_len+1)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len+1)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)

        spec_padded.zero_()
        refer1_padded.zero_()
        refer2_padded.zero_()
        wav_padded.zero_()
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            # phone, spec, refer, wav, tone, language 
            len_text = row[0].size(0)
            len_spec = row[1].size(1)
            len_refer1 = row[2].size(1)
            len_refer2 = row[3].size(1)
            len_wav = row[4].size(1)

            text_lengths[i] = len_text
            spec_lengths[i] = len_spec
            refer1_lengths[i] = len_refer1
            refer2_lengths[i] = len_refer2

            text_padded[i, :len_text] = row[0][:]
            spec_padded[i, :, :len_spec] = row[1][:]
            refer1_padded[i, :, :len_refer1] = row[2][:]
            refer2_padded[i, :, :len_refer2] = row[3][:]
            wav_padded[i, :, :len_wav] = row[4][:]
            tone_padded[i, :len_text] = row[5][:]
            language_padded[i, :len_text] = row[6][:]

        return text_padded, spec_padded, refer1_padded, refer2_padded, wav_padded, text_lengths, spec_lengths, refer1_lengths, refer2_lengths, tone_padded, language_padded

# cfg = json.load(open('./config.json'))
# train_dataset = TextAudioDataset(cfg)
# collate_fn = TextAudioCollate()
# dl = DataLoader(train_dataset, num_workers=0, shuffle=False, pin_memory=True,
#                 collate_fn=collate_fn)
# text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, mel_padded, tone_padded, language_padded = next(iter(dl))
