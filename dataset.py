from glob import glob
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
        self.audiopaths = glob(os.path.join(cfg['data']['training_files'], "**/*.wav"), recursive=True)
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
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            if self.use_mel_spec_posterior:
                # if os.path.exists(filename.replace(".wav", ".spec.pt")):
                #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
                #     spec = spec_to_mel_torch(
                #         torch.load(filename.replace(".wav", ".spec.pt")), 
                #         self.filter_length, self.n_mel_channels, self.sampling_rate,
                #         self.hparams.mel_fmin, self.hparams.mel_fmax)
                spec = mel_spectrogram_torch(audio_norm, self.filter_length,
                    self.n_mel_channels, self.sampling_rate, self.hop_length,
                    self.win_length, self.hparams.mel_fmin, self.hparams.mel_fmax, center=False)
            else:
                spec = spectrogram_torch(audio_norm, self.filter_length,
                    self.sampling_rate, self.hop_length, self.win_length,
                    center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        # print(text, word2ph,phone, tone, language_str)
        pold = phone
        w2pho = [i for i in word2ph]
        word2ph = [i for i in word2ph]
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        pold2 = phone

        if self.add_blank:
            p1 = len(phone)
            phone = commons.intersperse(phone, 0)
            p2 = len(phone)
            t1 = len(tone)
            tone = commons.intersperse(tone, 0)
            t2 = len(tone)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert = torch.load(bert_path)
            assert bert.shape[-1] == len(phone)
        except:
            bert = get_bert(text, word2ph, language_str)
            torch.save(bert, bert_path)
            #print(bert.shape[-1], bert_path, text, pold)
            assert bert.shape[-1] == len(phone)

        assert bert.shape[-1] == len(phone), (
        bert.shape, len(phone), sum(word2ph), p1, p2, t1, t2, pold, pold2, word2ph, text, w2pho)
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, phone, tone, language
    def get_audio_text_pair(self, audiopath, split='|'):
        text_path = audiopath.replace('.wav','.text')
        with open(text_path, encoding='utf-8') as f:
            texts = f.readline().strip().split(split)
        # separate filename, speaker_id and text
        language, text, phones, tone, word2ph = texts

        bert, phones, tone, language = self.get_text(text, word2ph, phones, tone, language, audiopath)

        spec, wav = self.get_audio(audiopath)
        return (phones, spec, wav, tone, language, bert)
    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths[index])

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]#B T C 

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].shape[1] for x in batch]),
            dim=0, descending=True)
        
        max_refer_len = max([x[0].size(1) for x in batch])
        max_f0_len = max([x[1].size(0) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])
        max_text_len = max([x[5].size(0) for x in batch])
        spec_dim = batch[0][2].shape[0]
        assert(max_f0_len == max_spec_len)

        spec_lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        f0_padded = torch.FloatTensor(len(batch), max_spec_len+1)
        spec_padded = torch.FloatTensor(len(batch), spec_dim, max_spec_len+1)
        refer_padded = torch.FloatTensor(len(batch), spec_dim, max_refer_len+1)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len+1)
        uv_padded = torch.FloatTensor(len(batch), max_spec_len+1)

        duration_padded = torch.LongTensor(len(batch), max_text_len+1)
        phoneme_padded = torch.LongTensor(len(batch), max_text_len+1)

        spec_padded.zero_()
        refer_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        phoneme_padded.zero_()
        duration_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            len_spec = row[2].size(1)
            len_text = row[5].size(0)
            len_refer = row[0].size(1)
            len_wav = row[3].size(1)
            spec_lengths[i] = len_spec
            refer_lengths[i] = len_refer
            text_lengths[i] = len_text
            # refer, f0, codes, audio, uv, phone, duration
            refer_padded[i, :, :len_refer] = row[0][:]
            f0_padded[i, :len_spec] = row[1][:]
            spec_padded[i, :, :len_spec] = row[2][:]
            wav_padded[i, :, :len_wav] = row[3][:]
            uv_padded[i, :len_spec] = row[4][:]
            phoneme_padded[i, :len_text] = row[5][:]
            duration_padded[i, :len_text] = row[6][:]

        return refer_padded, f0_padded, spec_padded, \
        wav_padded, spec_lengths, refer_lengths, text_lengths,\
        uv_padded, phoneme_padded, duration_padded
