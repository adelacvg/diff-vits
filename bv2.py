from collections import namedtuple
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
from einops import rearrange, reduce
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from vocos import Vocos
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torchaudio
from tqdm import tqdm

import commons
from dataset import TextAudioCollate, TextAudioDataset
from losses import kl_loss
from model import ConvLayer, LayerNorm, TransformerEncoderLayer, count_parameters, generate_index, group_hidden_by_segs
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from commons import init_weights, get_padding
from text import symbols, num_tones, num_languages
from unet1d.embeddings import TextTimeEmbedding
from unet1d.unet_1d_condition import UNet1DConditionModel
from utils import plot_spectrogram_to_numpy
import utils


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor1(nn.Module):
    def __init__(self,
        in_channels=512,
        hidden_channels=256,
        out_channels=1,
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
class DurationPredictor(nn.Module):
    def __init__(self,
        in_channels,
        hidden_channels,
        prompt_channels,
        kernel_size,
        p_dropout,
        out_channels=1,
        n_heads=8):
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
        self.prompt_proj = nn.Conv1d(prompt_channels, hidden_channels,1)
    # MultiHeadAttention 
    def forward(self, x, x_lengths, prompt, prompt_lengths):
        assert torch.isnan(x).any() == False
        x = x.detach()
        prompt = prompt.detach()
        prompt = self.prompt_proj(prompt)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lengths, prompt.size(2)), 1).to(x.dtype)
        # prompt_mask = ~commons.sequence_mask(prompt_lengths, prompt.size(2)).to(torch.bool)
        x = self.pre(x)*x_mask
        prompt = prompt*prompt_mask
        # prompt = prompt.masked_fill(prompt_mask.t().unsqueeze(-1), 0)
        assert torch.isnan(x).any() == False
        x = self.enc(x,1,prompt.transpose(1,2),encoder_attention_mask=prompt_mask).sample
        assert torch.isnan(x).any() == False
        x = x*x_mask
        return x
class DurationPredictor_legacy(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)

        self.arch = [8 for _ in range(n_layers)]
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], hidden_channels, p_dropout)
            for i in range(self.n_layers)
        ])

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, tone, language):
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x = rearrange(x, 'b c t->t b c')
        x_mask = ~commons.sequence_mask(x_lengths, x.size(0)).to(torch.bool)
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=x_mask)
        x = rearrange(x, 't b c -> b c t')
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
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
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
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
        x = rearrange(x, 't b c->b c t')
        return x
class Ph_Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
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
class Ph_p_encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 spec_channels,
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
            out_channels=hidden_channels,
            block_out_channels=(hidden_channels//4,hidden_channels//2,hidden_channels,hidden_channels),
            norm_num_groups=8,
            cross_attention_dim=hidden_channels,
            attention_head_dim=n_heads,
            addition_embed_type='text',
            resnet_time_scale_shift='scale_shift',
        )
        self.prompt_proj = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, refer, refer_lengths):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        refer_mask =  torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(x.dtype)
        refer = self.prompt_proj(refer)*refer_mask
        x = self.pre(x) * x_mask
        x = self.enc(x, 0, refer.transpose(1,2), encoder_attention_mask=refer_mask).sample
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
class VITS(nn.Module):
    def __init__(
        self,
        n_vocab,
        spec_channels,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=256,
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=6,
        flow_share_parameter=False,
        use_transformer_flow=True,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
            )
        # self.sdp = StochasticDurationPredictor(
        #     hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        # )
        self.dp = DurationPredictor(
            hidden_channels, 256,spec_channels, 3, 0.5
        )
        self.ref_enc = TextTimeEmbedding(spec_channels, gin_channels,1)
        self.ph_encoder_q = Ph_Encoder(inter_channels,inter_channels,inter_channels,3)
        self.phoneme_flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 6, gin_channels=gin_channels)
        self.ph_enc_p = Ph_p_encoder(hidden_channels,hidden_channels,inter_channels,spec_channels,3)
    def forward(self, x, x_lengths, y, y_lengths, tone, language):
        g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)

        logw_ = torch.log(w + 1e-6) * x_mask
        # prompt = self.prompt_proj(y)*y_mask
        logw = self.dp(x, x_lengths, y, y_lengths)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging

        l_length = l_length_dp 
        l_length = torch.sum(l_length.float())

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        #phoneme flow vae
        seg_ids = generate_index(w)
        z_q_ph, cnt_q_ph = group_hidden_by_segs(z.transpose(1,2), seg_ids, torch.max(x_lengths))
        z_q_ph, m_q_ph, logs_q_ph, _ = self.ph_encoder_q(z_q_ph.transpose(1,2), x_lengths)
        z_p_ph = self.phoneme_flow(z_q_ph, x_mask,g=g)
        ph_p, m_p_ph, logs_p_ph, _ = self.ph_enc_p(x, x_lengths, y, y_lengths)

        prosody = torch.zeros_like(z)
        for b in range(z.shape[0]):
            prosody[b,:,:y_lengths[b]] = z_q_ph[b].repeat_interleave(w[b,0].long(), dim=1)
        z = z + prosody
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        loss_kl_ph = kl_loss(z_p_ph, logs_q_ph, m_p_ph,logs_p_ph, x_mask)
        # loss_kl_ph = 0
        return z, y_lengths,(l_length, loss_kl, loss_kl_ph)

    def infer(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        tone,
        language,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
    ):
        g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, g=g
        )
        ph_p, m_p_ph, logs_p_ph, _ = self.ph_enc_p(x, x_lengths, y, y_lengths)
        ph_p = m_p + torch.randn_like(m_p_ph) * torch.exp(logs_p_ph) * noise_scale
        z_q_ph = self.phoneme_flow(ph_p, x_mask,g=g, reverse=True)

        logw =  self.dp(x, x_lengths, y, y_lengths)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)*y_mask

        prosody = torch.zeros_like(z)
        for b in range(z.shape[0]):
            prosody[b,:,:sum(w_ceil[b,0]).long()] = z_q_ph[b].repeat_interleave(w_ceil[b,0].long(), dim=1)
        z = z + prosody
        return z, y

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
        block_out_channels=(hidden_channels//2,hidden_channels//4*3,hidden_channels,hidden_channels),
        norm_num_groups=8,
        cross_attention_dim=hidden_channels,
        attention_head_dim=n_heads,
        addition_embed_type='text',
        resnet_time_scale_shift='scale_shift',
    )
    self.spec_channels = 513
    self.prompt_encoder = PromptEncoder(self.spec_channels, hidden_channels, hidden_channels,4,0.2)
  def forward(self, x, data, t):
    assert torch.isnan(x).any() == False
    cond, prompt, cond_lengths, prompt_lengths = data
    prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lengths, prompt.size(2)), 1).to( x.dtype)
    prompt = self.prompt_encoder(prompt, prompt_lengths)*prompt_mask
    x = torch.cat([x, cond], dim=1)

    # x_mask = commons.sequence_mask(contentvec_lengths, x.size(2)).to(torch.bool)
    prompt_mask = commons.sequence_mask(prompt_lengths, prompt.size(2)).to(torch.bool)
    x = self.unet(x, t, prompt.transpose(1,2), encoder_attention_mask=prompt_mask)

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
def exists(x):
    return x is not None
def cycle(dl):
    while True:
        for data in dl:
            yield data
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def rand_slice(z, spec, z_lengths, mel_padded):
    content_lengths = torch.zeros_like(z_lengths)
    prompt_lengths = torch.zeros_like(z_lengths)
    prompt = torch.zeros_like(spec)
    content = torch.zeros_like(z)
    x_start = torch.zeros_like(mel_padded)
    for i,len_z in enumerate(z_lengths):
        l = random.randint(int(len_z//3), int(len_z//3*2))
        u = random.randint(0, len_z-l)
        v = u + l
        if torch.rand(1)<0.5:
            content_lengths[i] = len_z-l
            prompt_lengths[i] = l
            prompt[i,:, :v-u] += spec[i, :, u:v]
            content[i,:,:len_z-v+u] += torch.cat([z[i,:, :u], z[i,:, v:len_z]], dim=-1)
            x_start[i,:,:len_z-v+u] += torch.cat([mel_padded[i,:, :u], mel_padded[i,:, v:len_z]], dim=-1)
        else:
            content_lengths[i] = l
            prompt_lengths[i]=len_z-l
            prompt[i,:,:len_z-v+u] += torch.cat([spec[i,:, :u], spec[i,:, v:len_z]], dim=-1) 
            content[i, :, :v-u] += z[i,:,u:v]
            x_start[i, :, :v-u] += mel_padded[i,:,u:v]          
    return content, content_lengths, prompt, prompt_lengths, x_start
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
        self.vits = VITS(
            len(symbols),
            cfg['data']['window_size'] // 2 + 1,
            mas_noise_scale_initial=0.01,
            noise_scale_delta=2e-6,
            **cfg['vits'])
        print('vits params: ', count_parameters(self.vits))
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
        content, refer, lengths = self.vits.infer(data)
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
        content, refer, lengths = self.vits.infer(data)
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
            content, refer = self.vits.infer(data)
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

            content, refer = self.vits.infer(text, text_lengths, spec, spec_lengths, tone, language)
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
        global step
        if self.vits.use_noise_scaled_mas:
            current_mas_noise_scale = (
                self.vits.mas_noise_scale_initial
                - self.vits.noise_scale_delta * step
            )
        self.vits.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        text_padded, text_lengths, spec_padded,\
        spec_lengths, wav_padded, wav_lengths, \
        mel_padded, tone_padded, language_padded = data
        b, d, n, device = *spec_padded.shape, spec_padded.device
        # get pre model outputs
        content, lengths, losses = self.vits(text_padded, text_lengths, spec_padded, spec_lengths, tone_padded, language_padded)
        content, lengths, refer, refer_lengths, x_start = rand_slice(content, spec_padded, lengths, mel_padded)

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, content.size(2)), 1).to(spec_padded.dtype)
        x_start = x_start*x_mask
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
        loss = 45*loss_diff + l_length + loss_kl + loss_kl_ph

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
        global step
        step = self.step
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
