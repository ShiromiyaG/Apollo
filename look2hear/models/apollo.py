import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, T)
        B, N, T = input.shape
        assert N % self.groups == 0

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)

class RMVN(nn.Module):
    """
    Rescaled MVN.
    """
    def __init__(self, dimension, groups=1):
        super(RMVN, self).__init__()

        self.mean = nn.Parameter(torch.zeros(dimension))
        self.std = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, *)
        B, N = input.shape[:2]
        assert N % self.groups == 0
        input_reshape = input.reshape(B, self.groups, N // self.groups, -1)
        T = input_reshape.shape[-1]

        input_norm = (input_reshape - input_reshape.mean(2).unsqueeze(2)) / (input_reshape.var(2).unsqueeze(2) + self.eps).sqrt()
        input_norm = input_norm.reshape(B, N, T) * self.std.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)

        return input_norm.reshape(input.shape)


def grouped_rms_norm(input, weight, eps=1e-5):
    """Batched equivalent of ``RMSNorm(groups=1)`` applied to a stack of bands.

    Args:
        input: (B, G, N, T) -- G independent bands, each normalised over its own N axis.
        weight: (G, N) -- the per-band ``RMSNorm.weight``.

    Returns the same values ``RMSNorm`` would produce band by band, including the
    cast back to the input dtype *before* the affine multiply.
    """
    input_float = input.float()
    input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + eps)

    return input_norm.type_as(input) * weight.unsqueeze(0).unsqueeze(-1)


class Roformer(nn.Module):
    """
    Transformer with rotary positional embedding.
    """
    def __init__(self, input_size, hidden_size, num_head=8, theta=10000, window=10000,
                 input_drop=0., attention_drop=0., causal=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.theta = theta  # base frequency for RoPE
        self.window = window
        # pre-calculate rotary embeddings
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)  # win, N
        self.register_buffer("sin_freq", sin_freq)  # win, N

        self.attention_drop = attention_drop
        self.causal = causal
        self.eps = 1e-5

        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size*self.num_head*3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size*self.num_head, self.input_size, 1, bias=False)

        self.MLP = nn.Sequential(RMSNorm(self.input_size),
                                 nn.Conv1d(self.input_size, self.input_size*8, 1, bias=False),
                                 nn.SiLU()
                                )
        self.MLP_output = nn.Conv1d(self.input_size*4, self.input_size, 1, bias=False)

    def _calc_rotary_emb(self):
        freq = 1. / (self.theta ** (torch.arange(0, self.hidden_size, 2)[:(self.hidden_size // 2)] / self.hidden_size))  # theta_i
        freq = freq.reshape(1, -1)  # 1, N//2
        pos = torch.arange(0, self.window).reshape(-1, 1)  # win, 1
        cos_freq = torch.cos(pos*freq)  # win, N//2
        sin_freq = torch.sin(pos*freq)  # win, N//2
        cos_freq = torch.stack([cos_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N
        sin_freq = torch.stack([sin_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N

        return cos_freq, sin_freq

    @staticmethod
    def _rotate_half_interleaved(feature):
        # (a, b) pairs along the last axis -> (-b, a); pure-torch so no host sync
        N = feature.shape[-1]
        pairs = feature.reshape(*feature.shape[:-1], N // 2, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)

        return rotated.reshape(feature.shape)

    def _add_rotary_emb(self, feature, pos):
        # feature shape: ..., N
        N = feature.shape[-1]

        feature_reshape = feature.reshape(-1, N)
        pos = min(pos, self.window-1)
        cos_freq = self.cos_freq[pos]
        sin_freq = self.sin_freq[pos]
        feature_reshape_neg = self._rotate_half_interleaved(feature_reshape)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)

        return feature_rope.reshape(feature.shape)

    def _add_rotary_sequence(self, feature):
        # feature shape: ..., T, N
        T, N = feature.shape[-2:]
        feature_reshape = feature.reshape(-1, T, N)

        cos_freq = self.cos_freq[:T]
        sin_freq = self.sin_freq[:T]
        feature_reshape_neg = self._rotate_half_interleaved(feature_reshape)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)

        return feature_rope.reshape(feature.shape)

    def forward(self, input):
        # input shape: B, N, T

        B, _, T = input.shape

        weight = self.weight(self.input_drop(self.input_norm(input))).reshape(B, self.num_head, self.hidden_size*3, T).mT
        Q, K, V = torch.split(weight, self.hidden_size, dim=-1)  # B, num_head, T, N

        # rotary positional embedding
        Q_rot = self._add_rotary_sequence(Q)
        K_rot = self._add_rotary_sequence(K)

        attention_output = F.scaled_dot_product_attention(Q_rot.contiguous(), K_rot.contiguous(), V.contiguous(), dropout_p=self.attention_drop, is_causal=self.causal)  # B, num_head, T, N
        attention_output = attention_output.mT.reshape(B, -1, T)
        output = self.output(attention_output) + input

        gate, z = self.MLP(output).chunk(2, dim=1)
        output = output + self.MLP_output(F.silu(gate) * z)

        return output, (K_rot, V)

class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super(ConvActNorm1d, self).__init__()

        self.in_channel = in_channel
        self.kernel = kernel
        self.causal = causal
        if not causal:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel-1)//2, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=kernel-1, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                     )

    def forward(self, input):

        output = self.conv(input)
        if self.causal:
            output = output[...,:-self.kernel+1]
        return input + output

class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super(ICB, self).__init__()

        self.blocks = nn.Sequential(ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal)
                                    )

    def forward(self, input):

        return self.blocks(input)

class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super(BSNet, self).__init__()

        self.feature_dim = feature_dim

        self.band_net = Roformer(self.feature_dim, self.feature_dim, num_head=8, window=100, causal=False)
        self.seq_net = ICB(self.feature_dim, kernel=kernel)

        # Rows of the band-attention input (B*T) and of the sequence model (B*nband)
        # are fully independent, so at inference we can stream them in slices to cap
        # peak memory without changing a single output value. 0 disables slicing.
        # Budget is expressed in float32 elements of the widest intermediate tensor.
        self.slice_elems = 0

    def _run_sliced(self, fn, x, elems_per_row):
        """Apply ``fn`` over dim 0 in slices sized to fit the element budget."""
        if self.slice_elems <= 0 or torch.is_grad_enabled():
            return fn(x)
        chunk = max(1, int(self.slice_elems // max(1, elems_per_row)))
        if x.shape[0] <= chunk:
            return fn(x)
        return torch.cat([fn(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], 0)

    def forward(self, input):
        # input shape: B, nband, N, T

        B, nband, N, T = input.shape

        # band comm -- widest intermediate is the Roformer MLP, 8x the feature dim
        band_input = input.permute(0,3,2,1).reshape(B*T, -1, nband)
        band_output = self._run_sliced(lambda t: self.band_net(t)[0], band_input,
                                       8 * self.feature_dim * nband)
        band_output = band_output.reshape(B, T, -1, nband).permute(0,3,2,1)

        # sequence modeling -- widest intermediate is the ConvActNorm1d hidden, 4x
        seq_input = band_output.reshape(B*nband, -1, T)
        output = self._run_sliced(self.seq_net, seq_input,
                                  4 * self.feature_dim * T).reshape(B, nband, -1, T)  # B, nband, N, T

        return output

class Apollo(BaseModel):
    """Band-split audio restoration model.

    The per-band input/output projections are stored as ``nn.ModuleList``s so that
    checkpoints stay byte-compatible with the reference implementation, but the
    forward pass runs them as batched matmuls over runs of equally sized bands.
    With the default 44.1 kHz / 20 ms setting that turns ~320 tiny kernel launches
    per forward into ~8, which is where most of the wall-clock time went.
    """

    def __init__(
        self,
        sr: int,
        win: int,
        feature_dim: int,
        layer: int
    ):
        super().__init__(sample_rate=sr)

        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # 80 bands
        bandwidth = int(self.win / 160)
        self.band_width = [bandwidth]*79
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(RMSNorm(self.band_width[i]*2+1),
                                         nn.Conv1d(self.band_width[i]*2+1, self.feature_dim, 1))
                          )

        self.net = []
        for _ in range(layer):
            self.net.append(BSNet(self.feature_dim))
        self.net = nn.Sequential(*self.net)

        self.output = nn.ModuleList([])
        for i in range(self.nband):
            self.output.append(nn.Sequential(RMSNorm(self.feature_dim),
                                                 nn.Conv1d(self.feature_dim, self.band_width[i]*4, 1),
                                                 nn.GLU(dim=1)
                                                )
                                  )

        # Consecutive bands of equal width share a projection shape and can be run
        # as one batched matmul. Stored as (first_band, n_bands, width) triples.
        self.band_groups = self._build_band_groups(self.band_width)
        # STFT window is fixed for the lifetime of the model; keep it off the
        # allocator hot path. persistent=False keeps it out of the state dict.
        self.register_buffer("stft_window", torch.hann_window(self.win), persistent=False)
        self._fused_cache = None
        self.grad_checkpointing = False

    def set_gradient_checkpointing(self, enabled=True):
        """Recompute each band-split block during the backward pass.

        Training activations dominate VRAM here: every one of the `layer` blocks
        holds a (B*nch, nband, feature_dim, T) tensor plus the 8x-wider Roformer MLP
        on top of it. Checkpointing keeps only the block inputs and rebuilds the
        rest on the way back, which is what makes fine-tuning fit on 8 GB. Costs
        roughly one extra forward pass of compute; the result is exact.
        """
        self.grad_checkpointing = bool(enabled)
        return self

    def set_vram_budget(self, megabytes):
        """Cap peak inference memory by streaming the band/sequence blocks in slices.

        Rows of both blocks are independent, so this is bit-exact -- it trades a few
        extra kernel launches for a peak-memory ceiling that no longer scales with
        clip length. ``megabytes`` bounds the widest single intermediate tensor, not
        total allocation; the realistic floor is a few hundred MB of activations that
        must stay live regardless. Pass 0 to disable. No effect while grad is enabled.
        """
        elems = int(float(megabytes) * (2 ** 20) / 4) if megabytes else 0
        for module in self.modules():
            if isinstance(module, BSNet):
                module.slice_elems = elems
        return self

    @staticmethod
    def _build_band_groups(band_width):
        groups = []
        start = 0
        for i in range(1, len(band_width) + 1):
            if i == len(band_width) or band_width[i] != band_width[start]:
                groups.append((start, i - start, band_width[start]))
                start = i
        return groups

    # ------------------------------------------------------------------
    # fused per-band projection weights
    # ------------------------------------------------------------------
    def _stack_fused_weights(self):
        """Stack the per-band projection parameters into per-group tensors."""
        fused = []
        for start, count, width in self.band_groups:
            bands = range(start, start + count)
            fused.append({
                "bn_norm": torch.stack([self.BN[i][0].weight for i in bands]),                    # (G, 2w+1)
                "bn_weight": torch.stack([self.BN[i][1].weight.squeeze(-1) for i in bands]),      # (G, D, 2w+1)
                "bn_bias": torch.stack([self.BN[i][1].bias for i in bands]),                      # (G, D)
                "out_norm": torch.stack([self.output[i][0].weight for i in bands]),               # (G, D)
                "out_weight": torch.stack([self.output[i][1].weight.squeeze(-1) for i in bands]), # (G, 4w, D)
                "out_bias": torch.stack([self.output[i][1].bias for i in bands]),                 # (G, 4w)
            })
        return fused

    def _get_fused_weights(self):
        # In train mode the parameters change every step, so restack each call
        # (one cat kernel per tensor -- still far cheaper than the band loop).
        if self.training or torch.is_grad_enabled():
            return self._stack_fused_weights()
        if self._fused_cache is None:
            self._fused_cache = self._stack_fused_weights()
        return self._fused_cache

    def _invalidate_fused_cache(self):
        self._fused_cache = None

    def train(self, mode=True):
        self._invalidate_fused_cache()
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        # covers .cuda() / .to() / .half() moving or recasting the parameters
        self._invalidate_fused_cache()
        return super()._apply(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._invalidate_fused_cache()
        return result

    # ------------------------------------------------------------------

    def spec_band_split(self, input):

        B, nch, nsample = input.shape

        spec = torch.stft(input.reshape(B*nch, nsample), n_fft=self.win, hop_length=self.stride,
                          window=self.stft_window.to(dtype=input.dtype, device=input.device),
                          return_complex=True)

        subband_spec = []
        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(self.nband):
            this_spec = spec[:,band_idx:band_idx+self.band_width[i]]
            subband_spec.append(this_spec)  # B, BW, T
            subband_power.append((this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1))  # B, 1, T
            subband_spec_norm.append(torch.complex(this_spec.real / subband_power[-1], this_spec.imag / subband_power[-1]))  # B, BW, T
            band_idx += self.band_width[i]
        subband_power = torch.cat(subband_power, 1)  # B, nband, T

        return subband_spec_norm, subband_power

    def feature_extractor(self, input):

        subband_spec_norm, subband_power = self.spec_band_split(input)

        # normalization and bottleneck
        subband_feature = []
        for i in range(self.nband):
            concat_spec = torch.cat([subband_spec_norm[i].real, subband_spec_norm[i].imag, torch.log(subband_power[:,i].unsqueeze(1))], 1)
            subband_feature.append(self.BN[i](concat_spec))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        return subband_feature

    def _fused_feature_extractor(self, spec, fused):
        """Batched replacement for ``spec_band_split`` + ``feature_extractor``.

        ``spec`` is the complex STFT of shape (B*nch, enc_dim, T).
        """
        features = []
        bin_idx = 0
        for (start, count, width), weights in zip(self.band_groups, fused):
            nbins = count * width
            band = spec[:, bin_idx:bin_idx + nbins]
            bin_idx += nbins
            Bc, _, T = band.shape
            real = band.real.reshape(Bc, count, width, T)
            imag = band.imag.reshape(Bc, count, width, T)

            power = (real.pow(2).sum(2) + imag.pow(2).sum(2) + self.eps).sqrt().unsqueeze(2)  # (Bc, G, 1, T)
            concat_spec = torch.cat([real / power, imag / power, torch.log(power)], 2)        # (Bc, G, 2w+1, T)

            normed = grouped_rms_norm(concat_spec, weights["bn_norm"])
            feat = torch.matmul(weights["bn_weight"], normed) + weights["bn_bias"].unsqueeze(0).unsqueeze(-1)
            features.append(feat)

        return torch.cat(features, 1)  # (Bc, nband, D, T)

    def _fused_output(self, feature, fused):
        """Batched replacement for the per-band output heads. Returns the complex spectrum."""
        est_spec = []
        band_idx = 0
        for (start, count, width), weights in zip(self.band_groups, fused):
            feat = feature[:, band_idx:band_idx + count]  # (Bc, G, D, T)
            band_idx += count

            normed = grouped_rms_norm(feat, weights["out_norm"])
            proj = torch.matmul(weights["out_weight"], normed) + weights["out_bias"].unsqueeze(0).unsqueeze(-1)
            RI = F.glu(proj, dim=2)  # (Bc, G, 2w, T)

            Bc, _, _, T = RI.shape
            # torch.complex rejects reduced precision, so pin this back to fp32;
            # without it the model cannot run under autocast at all.
            RI = RI.float().reshape(Bc, count, 2, width, T)
            est_spec.append(torch.complex(RI[:, :, 0], RI[:, :, 1]).reshape(Bc, count * width, T))

        return torch.cat(est_spec, 1)

    def forward(self, input):

        B, nch, nsample = input.shape

        fused = self._get_fused_weights()

        spec = torch.stft(input.reshape(B*nch, nsample), n_fft=self.win, hop_length=self.stride,
                          window=self.stft_window.to(dtype=input.dtype, device=input.device),
                          return_complex=True)

        subband_feature = self._fused_feature_extractor(spec, fused)

        if self.grad_checkpointing and torch.is_grad_enabled():
            feature = subband_feature
            for block in self.net:
                feature = torch.utils.checkpoint.checkpoint(block, feature, use_reentrant=False)
        else:
            feature = self.net(subband_feature)

        est_spec = self._fused_output(feature, fused)

        output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride,
                             window=self.stft_window.to(dtype=est_spec.real.dtype, device=input.device),
                             length=nsample).reshape(B, nch, -1)

        return output

    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args
