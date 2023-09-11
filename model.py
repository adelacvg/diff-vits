import abc
import random
from datetime import datetime
from matplotlib import pyplot as plt
from losses import kl_loss
import monotonic_align
from unet1d.unet_1d_condition import UNet1DConditionModel
from unet1d.embeddings import TextTimeEmbedding
from vocos import Vocos
import json
import os
from pathlib import Path
from utils import plot_spectrogram_to_numpy
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from operations import OPERATIONS_ENCODER, MultiheadAttention, SinusoidalPositionalEmbedding, TransformerFFNLayer
from parametrizations import weight_norm
from torch import nn
import torchaudio
from dataset import TextAudioCollate, TextAudioDataset
import commons
import modules
from accelerate import Accelerator
from ema_pytorch import EMA
from accelerate import DistributedDataParallelKwargs
import math
from pathlib import Path
from modules import ResidualCouplingLayer
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules import Module

from text.symbols import symbols, num_tones, num_languages
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import utils
from torch.cuda.amp import GradScaler

from tqdm.auto import tqdm
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def exists(x):
    return x is not None
def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

class TransformerEncoderLayer(nn.Module):
    def __init__(self, layer, hidden_size, dropout):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

class TextEncoder(nn.Module):
    def __init__(self,
      hidden_channels=512,
      out_channels=512,
      n_layers=6,
      p_dropout=0.2
      ):
        super().__init__()
        self.arch = [8 for _ in range(n_layers)]
        self.num_layers = n_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.padding_idx = 0
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels ** -0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels ** -0.5)
        self.dropout = p_dropout
        self.embed_scale = math.sqrt(hidden_channels)
        self.max_source_positions = 2000
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_channels, self.dropout)
            for i in range(self.num_layers)
        ])
        self.proj = nn.Conv1d(hidden_channels, out_channels*2, 1)

    def forward(self, text_padded, text_lengths, tone_padded, language_padded):
        assert torch.isnan(self.emb.weight).any() == False
        x = (self.emb(text_padded)+ self.tone_emb(tone_padded)+ self.language_emb(language_padded)) * math.sqrt(self.hidden_channels)  # [b, t, h]
        assert torch.isnan(x).any() == False

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        x_mask = ~commons.sequence_mask(text_lengths, x.size(0)).to(torch.bool)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=x_mask)

        assert torch.isnan(x).any() == False
        x = rearrange(x, 't b c->b c t')
        x_mask = torch.unsqueeze(commons.sequence_mask(text_lengths, x.size(2)), 1).to(x.dtype)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dropout=0):
        super().__init__()
        self.layer_norm = LayerNorm(c_in)
        conv = ConvTBC(c_in, c_out, kernel_size, padding=kernel_size // 2)
        self.conv = conv
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c_in))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        return x

class EncConvLayer(nn.Module):
    def __init__(self, c, kernel_size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(c)
        conv = ConvTBC(c, c, kernel_size, padding=kernel_size // 2)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = weight_norm(conv, dim=2)
        self.dropout = dropout

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        residual = x
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x + residual
        return x
class PromptEncoder(nn.Module):
    def __init__(self,
      in_channels=128,
      hidden_channels=512,
      out_channels=128,
      n_layers=6,
      p_dropout=0.2,
      last_ln = True,):
        super().__init__()
        self.arch = [8 for _ in range(n_layers)]
        self.num_layers = n_layers
        self.hidden_size = hidden_channels
        self.padding_idx = 0
        self.dropout = p_dropout
        self.embed_scale = math.sqrt(hidden_channels)
        self.max_source_positions = 2000
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(out_channels)
        self.pre = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        self.out_proj = ConvLayer(hidden_channels, out_channels, 1)

    def forward(self, src_tokens, lengths=None):
        # B x C x T -> T x B x C
        src_tokens = rearrange(src_tokens, 'b c t -> t b c')
        # compute padding mask
        encoder_padding_mask = ~commons.sequence_mask(lengths, src_tokens.size(0)).to(torch.bool)
        x = src_tokens

        x = self.pre(x, encoder_padding_mask=encoder_padding_mask)
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        x = self.out_proj(x) * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
class Ph_p_encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = UNet1DConditionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=(hidden_channels//4,hidden_channels//2,hidden_channels,hidden_channels),
            norm_num_groups=8,
            cross_attention_dim=hidden_channels,
            attention_head_dim=n_heads,
            addition_embed_type='text',
            resnet_time_scale_shift='scale_shift',
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, refer, refer_lengths):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        refer_mask =  torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, 0, refer.transpose(1,2), encoder_attention_mask=refer_mask).sample
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
class SpecEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = UNet1DConditionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=(hidden_channels//4,hidden_channels//2,hidden_channels,hidden_channels),
            norm_num_groups=8,
            cross_attention_dim=hidden_channels,
            attention_head_dim=n_heads,
            addition_embed_type='text',
            resnet_time_scale_shift='scale_shift',
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, 0, x.transpose(1,2), encoder_attention_mask=x_mask).sample
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
class Ph_Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        # x = self.enc(x, 0, x.transpose(1,2), encoder_attention_mask=x_mask).sample
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
class DurationPredictor(nn.Module):
    def __init__(self,
        in_channels=512,
        hidden_channels=256,
        out_channels=1,
        attention_layers=10,
        n_heads=8,
        p_dropout=0.5,):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.act = nn.ModuleList()
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.enc = UNet1DConditionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=(hidden_channels//4,hidden_channels//2,hidden_channels,hidden_channels),
            norm_num_groups=8,
            cross_attention_dim=hidden_channels,
            attention_head_dim=n_heads,
            addition_embed_type='text',
            resnet_time_scale_shift='scale_shift',
        )
    # MultiHeadAttention 
    def forward(self, x, x_lengths, prompt, prompt_lengths):
        assert torch.isnan(x).any() == False
        x = x.detach()
        prompt = prompt.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lengths, prompt.size(2)), 1).to(x.dtype)
        # prompt_mask = ~commons.sequence_mask(prompt_lengths, prompt.size(2)).to(torch.bool)
        x = self.pre(x)*x_mask
        prompt = prompt*prompt_mask
        # prompt = prompt.masked_fill(prompt_mask.t().unsqueeze(-1), 0)
        assert torch.isnan(x).any() == False
        x = self.enc(x,0,prompt.transpose(1,2),encoder_attention_mask=prompt_mask).sample
        assert torch.isnan(x).any() == False
        x = x*x_mask
        return x
def group_hidden_by_segs(h, seg_ids, max_len):
    """
    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    return h_gby_segs, cnt_gby_segs
def generate_index(w):
    B,_, T = w.shape
    lens = w.sum(dim=2,keepdim=True).squeeze(1)
    max_len = torch.max(lens).item()

    index_tensor = torch.zeros((B,int(max_len)), dtype=torch.long).to(w.device)
    for b in range(B):
        index = torch.arange(1,T+1).to(w.device)
        index_tensor[b,:int(lens[b][0])] = index.repeat_interleave(w[b,0].long())

    return index_tensor
def rand_slice_prompt(spec,l,u,v):
    max_len = max(l)
    prompt = torch.zeros(spec.shape[0],spec.shape[1],max_len)
    for i in range(spec.shape[0]):
        prompt[i,:,:l[i]] = spec[i,:,u[i]:v[i]]
    return prompt


class Pre_model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.phoneme_encoder = TextEncoder(**self.cfg['phoneme_encoder'])
        print("phoneme params:", count_parameters(self.phoneme_encoder))
        self.prompt_encoder = PromptEncoder(**self.cfg['prompt_encoder'])
        print("prompt params:", count_parameters(self.prompt_encoder))
        self.spec_encoder = SpecEncoder(**self.cfg['spec_encoder'])
        print("spec params:", count_parameters(self.prompt_encoder))
        self.spec_flow = ResidualCouplingBlock(**self.cfg['spec_flow'])
        print("spec flow params:", count_parameters(self.prompt_encoder))
        self.dp = DurationPredictor(**self.cfg['duration_predictor'])
        print("dp params:", count_parameters(self.prompt_encoder))
        self.phoneme_flow = ResidualCouplingBlock(**self.cfg['phoneme_flow'])
        print("phoneme flow params:", count_parameters(self.phoneme_flow))
        self.ph_enc_p = Ph_p_encoder(**self.cfg['ph_p_encoder'])
        print("ph p encoder params:", count_parameters(self.ph_enc_p))
        self.attn_pooling = TextTimeEmbedding(self.cfg['spec_encoder']['in_channels'],
                                              self.cfg['spec_flow']['gin_channels'],8)
        self.spec_proj = nn.Conv1d(self.cfg['data']['window_size']//2+1,
                                   self.cfg['spec_encoder']['in_channels'],1)
        self.ph_encoder_q = Ph_Encoder(**self.cfg['ph_encoder'])
        self.use_noise_scaled_mas = self.cfg['train']['use_noise_scaled_mas']
        self.mas_noise_scale_initial = self.cfg['train']['mas_noise_scale_initial']
        self.noise_scale_delta = self.cfg['train']['noise_scale_delta']
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        
    def forward(self,data):
        text_padded, text_lengths, spec_padded,\
        spec_lengths, wav_padded, wav_lengths,\
        mel_padded, tone_padded, language_padded = data
        spec_padded = self.spec_proj(spec_padded)
        g = self.attn_pooling(spec_padded.transpose(1,2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.phoneme_encoder(text_padded, text_lengths, tone_padded, language_padded)
        z, m_q, logs_q, y_mask = self.spec_encoder(spec_padded, spec_lengths)
        z_p = self.spec_flow(z, y_mask, g=g)
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2),
                                     s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
               epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale
               neg_cent = neg_cent + epsilon
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
        w = attn.sum(2)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, text_lengths, spec_padded, spec_lengths)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging
        
        #phoneme flow vae
        seg_ids = generate_index(w)
        z_q_ph, cnt_q_ph = group_hidden_by_segs(z.transpose(1,2), seg_ids, torch.max(text_lengths))
        z_q_ph, m_q_ph, logs_q_ph, _ = self.ph_encoder_q(z_q_ph.transpose(1,2), text_lengths)
        z_p_ph = self.phoneme_flow(z_q_ph, x_mask,g=g)
        ph_p, m_p_ph, logs_p_ph, _ = self.ph_enc_p(x, text_lengths, spec_padded, spec_lengths)

        prosody = torch.zeros_like(z)
        for b in range(z.shape[0]):
            prosody[b,:,:spec_lengths[b]] = z_q_ph[b].repeat_interleave(w[b,0].long(), dim=1)
        z = z + prosody

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        l = [random.randint(int(sl//3), int(sl//3*2)) for sl in spec_lengths]
        u = [random.randint(0, spec_lengths[i]-l[i]) for i in range(spec_lengths.shape[0])]
        v = [u[i]+l[i] for i in range(spec_lengths.shape[0])]
        prompt = rand_slice_prompt(spec_padded, l, u, v).to(spec_padded.device)
        prompt_lengths=torch.Tensor(l).to(spec_lengths.device)
        prompt = self.prompt_encoder(normalize(prompt),prompt_lengths)

        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        loss_kl_ph = kl_loss(z_p_ph, logs_q_ph, m_p_ph,logs_p_ph, x_mask)
        l_length = torch.sum(l_length_dp.float())
        return z, spec_lengths, prompt, prompt_lengths, (l_length, loss_kl, loss_kl_ph)
    def infer(self, data, length_scale=1.0, noise_scale=.667):

        x, x_lengths, spec_padded,\
        spec_lengths, tone, language = data
        spec_padded = self.spec_proj(spec_padded)
        g = self.attn_pooling(spec_padded.transpose(1,2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.phoneme_encoder(x, x_lengths, tone, language)

        ph_p, m_p_ph, logs_p_ph, _ = self.ph_enc_p(x, x_lengths, spec_padded, spec_lengths)
        ph_p = m_p + torch.randn_like(m_p_ph) * torch.exp(logs_p_ph) * noise_scale
        z_q_ph = self.phoneme_flow(ph_p, x_mask,g=g, reverse=True)

        #length regulator
        logw = self.dp(x, x_lengths, spec_padded, spec_lengths)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.spec_flow(z_p, y_mask, g=g, reverse=True)

        prosody = torch.zeros_like(z)
        for b in range(z.shape[0]):
            prosody[b,:,:sum(w_ceil[b,0]).long()] = z_q_ph[b].repeat_interleave(w_ceil[b,0].long(), dim=1)
        z = z + prosody

        prompt = self.prompt_encoder(normalize(spec_padded),spec_lengths)
        return z, prompt

def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer

class Diffusion_Encoder(nn.Module):
  def __init__(self,
      in_channels=128,
      out_channels=128,
      hidden_channels=256,
      kernel_size=3,
      dilation_rate=2,
      n_layers=40,
      n_heads=8,
      p_dropout=0.2,
      dim_time_mult=None,
      ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_heads=n_heads
    self.unet = UNet1DConditionModel(
        in_channels=in_channels+hidden_channels,
        out_channels=out_channels,
        block_out_channels=(128,192,256,256),
        norm_num_groups=8,
        cross_attention_dim=hidden_channels,
        attention_head_dim=n_heads,
        addition_embed_type='text',
        resnet_time_scale_shift='scale_shift',
    )
  def forward(self, x, data, t):
    assert torch.isnan(x).any() == False
    cond, prompt, cond_lengths, prompt_lengths = data
    prompt = rearrange(prompt, 't b c -> b t c')
    x = torch.cat([x, cond], dim=1)

    # x_mask = commons.sequence_mask(contentvec_lengths, x.size(2)).to(torch.bool)
    prompt_mask = commons.sequence_mask(prompt_lengths, prompt.size(1)).to(torch.bool)
    x = self.unet(x, t, prompt, encoder_attention_mask=prompt_mask)

    return x.sample

# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))
# noise schedules

def normalize(code):
    # code = 10*torch.log10(1+code/100)
    # code = code/10
    return code
def denormalize(code):
    # code = 10*(10**(code/10)-1)
    # code = code*10
    return code

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
class NaturalSpeech2(nn.Module):
    def __init__(self,
        cfg,
        rvq_cross_entropy_loss_weight = 0.1,
        diff_loss_weight = 1.0,
        f0_loss_weight = 1.0,
        duration_loss_weight = 1.0,
        ddim_sampling_eta = 0,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
        ):
        super().__init__()
        self.pre_model = Pre_model(cfg)
        self.diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
        print("diff params: ", count_parameters(self.diff_model))
        self.dim = self.diff_model.in_channels
        timesteps = cfg['train']['timesteps']

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = timesteps

        self.sampling_timesteps = None
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.duration_loss_weight = duration_loss_weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr)
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data = None, clip_x_start = False, rederive_pred_noise = False):
        # x = rearrange(x, 'b c t -> t b c')
        model_output = self.diff_model(x,data, t)

        x_start = model_output
        # x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, data):
        preds = self.model_predictions(x, t, data)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, data):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, data=data)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, text, refer, text_lengths, refer_lengths):
        data = (text, refer, text_lengths, refer_lengths)
        content, refer, lengths = self.pre_model.infer(data)
        shape = (text.shape[0], self.dim, int(lengths.max().item()))
        batch, device = shape[0], refer.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, (content,refer,lengths,refer_lengths))
            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def ddim_sample(self, text, refer, text_lengths, refer_lengths):
        data = (text, refer, text_lengths, refer_lengths)
        content, refer, lengths = self.pre_model.infer(data)
        shape = (text.shape[0], self.dim, int(lengths.max().item()))
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, (content,refer,lengths,refer_lengths), rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = img
        return ret
    def sample_fun(self, x, t, data = None):
        model_output = self.diff_model(x,data, t)
        t = t.type(torch.int64) 
        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return x_start
    @torch.no_grad()
    def sample(self, text, spec, text_lengths, spec_lengths, tone, language, vocos, sampling_timesteps = 200, sample_method = 'unipc'):
        self.sampling_timesteps = sampling_timesteps
        if sample_method == 'ddpm':
            sample_fn = self.p_sample_loop
        elif sample_method == 'ddim':
            sample_fn = self.ddim_sample
        elif sample_method == 'dpmsolver':
            from sampler.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            def my_wrapper(fn):
                def wrapped(x, t, **kwargs):
                    ret = fn(x, t, **kwargs)
                    self.bar.update(1)
                    return ret

                return wrapped

            data = (text, spec, text_lengths, spec_lengths, tone, language)
            content, refer = self.pre_model.infer(data)
            shape = (content.shape[1], self.dim, content.shape[0])
            batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
            audio = torch.randn(shape, device = device)
            model_fn = model_wrapper(
                my_wrapper(self.sample_fun),
                noise_schedule,
                model_type="x_start",  #"noise" or "x_start" or "v" or "score"
                model_kwargs={"data":(content,refer,text_lengths,spec_lengths)}
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

            steps = 40
            self.bar = tqdm(desc="sample time step", total=steps)
            audio = dpm_solver.sample(
                audio,
                steps=steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            self.bar.close()
        elif sample_method =='unipc':
            from sampler.uni_pc import NoiseScheduleVP, model_wrapper, UniPC
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

            def my_wrapper(fn):
                def wrapped(x, t, **kwargs):
                    ret = fn(x, t, **kwargs)
                    self.bar.update(1)
                    return ret

                return wrapped

            data = (text, text_lengths, spec, spec_lengths, tone, language)
            content, refer = self.pre_model.infer(data)
            shape = (content.shape[0], self.dim, content.shape[2])
            batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
            audio = torch.randn(shape, device = device)
            model_fn = model_wrapper(
                my_wrapper(self.sample_fun),
                noise_schedule,
                model_type="x_start",  #"noise" or "x_start" or "v" or "score"
                model_kwargs={"data":(content,refer,text_lengths,spec_lengths)}
            )
            uni_pc = UniPC(model_fn, noise_schedule, variant='bh2')
            steps = 30
            self.bar = tqdm(desc="sample time step", total=steps)
            audio = uni_pc.sample(
                audio,
                steps=steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            self.bar.close()

        audio = denormalize(audio)
        mel = audio
        vocos.to(audio.device)
        audio = vocos.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio,mel 

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, data):
        text_padded, text_lengths, spec_padded,\
        spec_lengths, wav_padded, wav_lengths, \
        mel_padded, tone_padded, language_padded = data
        b, d, n, device = *spec_padded.shape, spec_padded.device
        x_mask = torch.unsqueeze(commons.sequence_mask(spec_lengths, spec_padded.size(2)), 1).to(spec_padded.dtype)
        x_start = normalize(mel_padded)*x_mask
        # get pre model outputs
        content, lengths, refer, refer_lengths, losses = self.pre_model(data)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = torch.randn_like(x_start)*x_mask
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        # predict and take gradient step
        model_out = self.diff_model(x,(content,refer,lengths,refer_lengths), t)
        target = x_start

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss_diff = reduce(loss, 'b ... -> b (...)', 'mean')
        loss_diff = loss_diff * extract(self.loss_weight, t, loss.shape)
        loss_diff = loss_diff.mean()

        l_length, loss_kl, loss_kl_ph = losses
        loss = loss_diff + l_length + loss_kl + loss_kl_ph

        # cross entropy loss to codebooks
        # _, indices, _, quantized_list = encode(codes_padded,8,codec)
        # ce_loss = rvq_ce_loss(denormalize(model_out.unsqueeze(0))-quantized_list, indices, codec)
        # loss = loss + 0.1 * ce_loss

        return loss, loss_diff, l_length, loss_kl, loss_kl_ph, model_out, target

def save_audio(audio, path, codec):
    audio = denormalize(audio)
    audio = audio.unsqueeze(0).transpose(1,2)
    audio = codec.decode(audio)
    if audio.ndim == 3:
        audio = rearrange(audio, 'b 1 n -> b n')
    audio = audio.detach().cpu()

    torchaudio.save(path, audio, 24000)
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2) 
    return total_norm
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
class Trainer(object):
    def __init__(
        self,
        cfg_path = './config.json',
    ):
        super().__init__()

        self.cfg = json.load(open(cfg_path))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        self.device = self.accelerator.device

        # model

        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.model = NaturalSpeech2(cfg=self.cfg).to(self.device)

        self.save_and_sample_every = self.cfg['train']['save_and_sample_every']

        self.batch_size = self.cfg['train']['train_batch_size']
        self.gradient_accumulate_every = self.cfg['train']['gradient_accumulate_every']

        self.train_num_steps = self.cfg['train']['train_num_steps']

        # dataset and dataloader
        train_dataset = TextAudioDataset(self.cfg)
        collate_fn = TextAudioCollate()
        self.ds = train_dataset
        dl = DataLoader(train_dataset,batch_size=self.cfg['train']['train_batch_size'], num_workers=self.cfg['train']['num_workers'], shuffle=False, pin_memory=True, collate_fn=collate_fn)
        self.dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        
        # optimizer
        self.opt = AdamW(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])
        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.eval_dl = DataLoader(train_dataset, batch_size = 1, shuffle = False, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)
            self.eval_dl = iter(cycle(self.eval_dl))
        now = datetime.now()
        self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
        self.logs_folder.mkdir(exist_ok = True, parents=True)
        # step counter state
        self.step = 0
        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(model_path, map_location=device)

        self.step = data['step']

        saved_state_dict = data['model']
        model = self.accelerator.unwrap_model(self.model)
        # new_state_dict= {}
        # for k,v in saved_state_dict.items():
        #     name=k[7:]
        #     new_state_dict[name] = v
        # if hasattr(model, 'module'):
        #     model.module.load_state_dict(new_state_dict)
        # else:
        model.load_state_dict(saved_state_dict)


    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        accelerator = self.accelerator
        device = self.device

        if accelerator.is_main_process:
            logger = utils.get_logger(self.logs_folder)
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = [d.to(device) for d in data]

                    with self.accelerator.autocast():
                        loss, loss_diff, loss_l, loss_kl, loss_kl_ph, pred, target = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                grad_norm = get_grad_norm(self.model)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                # for name, param in self.model.named_parameters():
                #     if torch.isnan(param.grad).any():
                #         print("nan gradient found", name)
                #         raise SystemExit
                self.opt.zero_grad()

                accelerator.wait_for_everyone()
############################logging#############################################
                if accelerator.is_main_process and self.step % 100 == 0:
                    logger.info('Train Epoch: {} [{:.0f}%]'.format(
                        self.step//len(self.ds),
                        100. * self.step / self.train_num_steps))
                    logger.info(f"Losses: {[loss_diff, loss_kl, loss_l, loss_kl_ph]}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss_diff, 
                                "loss/all": total_loss,
                                "loss/len":loss_l,
                                "loss/kl":loss_kl,
                                "loss/kl_ph":loss_kl_ph,
                                "loss/grad": grad_norm}
                    image_dict = {
                        "all/spec": plot_spectrogram_to_numpy(target[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred": plot_spectrogram_to_numpy(pred[0, :, :].detach().unsqueeze(-1).cpu()),
                    }

                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )
                self.step += 1
                if accelerator.is_main_process:
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        data = next(self.eval_dl)
                        data = [d.to(device) for d in data]
                        text, text_lengths, spec,\
                        spec_lengths, wav_padded, wav_lengths,\
                        mel_padded, tone, language = data
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            samples, mel = self.model.sample(text, spec, text_lengths, spec_lengths,\
                                tone, language, self.vocos)
                            samples = samples.detach().cpu()
                            

                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), samples, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": samples,
                                f"gt/audio": wav_padded[0]
                            })
                        image_dict = {
                            f"gen/mel":plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu()),
                            f"gt/mel":plot_spectrogram_to_numpy(mel_padded[0, :, :].detach().unsqueeze(-1).cpu()),
                        }
                        utils.summarize(
                            writer=writer_eval,
                            global_step=self.step,
                            audios=audio_dict,
                            images=image_dict,
                            audio_sampling_rate=24000
                        )
                        keep_ckpts = self.cfg['train']['keep_ckpts']
                        if keep_ckpts > 0:
                            utils.clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                        self.save(milestone)
                pbar.update(1)

        print('training complete')
