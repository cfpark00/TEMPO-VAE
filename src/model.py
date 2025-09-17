"""
Complete VAE model - EXACT mltools architecture in one file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============= Helper Functions =============

@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module


def get_conv(in_channels, out_channels, **kwargs):
    def_params = {
        "dim": 2,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1,
        "padding_mode": "zeros",
        "dilation": 1,
        "groups": 1,
        "init": lambda x: x,
        "transposed": False,
    }
    def_params.update(kwargs)
    dim = def_params.pop("dim")
    transposed = def_params.pop("transposed")
    init = def_params.pop("init")
    if dim == 2:
        conv = nn.ConvTranspose2d if transposed else nn.Conv2d
        return init(conv(in_channels, out_channels, **def_params))
    elif dim == 3:
        conv = nn.ConvTranspose3d if transposed else nn.Conv3d
        return init(conv(in_channels, out_channels, **def_params))


# ============= Distribution =============

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, dim=2, deterministic=False):
        self.parameters = parameters
        self.dim = dim
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=([1, 2, 3] if self.dim == 2 else [1, 2, 3, 4]),
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=([1, 2, 3] if self.dim == 2 else [1, 2, 3, 4]),
                )

    def mode(self):
        return self.mean


# ============= Building Blocks =============

class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=4, dim=2, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        assert (
            self.in_channels % n_heads == 0
        ), "in_channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.dim = dim
        assert self.dim == 2 or self.dim == 3, "dim must be 2 or 3"

        norm_params = kwargs.get("norm_params", {})

        self.norm = nn.GroupNorm(num_channels=in_channels, **norm_params)

        self.q = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.k = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.v = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dim == 2:
            b, c, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, h * w)
            k = k.reshape(b, c_, self.n_heads, h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, h, w)
            h_ = self.proj_out(h_)
        elif self.dim == 3:
            b, c, d, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, d * h * w)
            k = k.reshape(b, c_, self.n_heads, d * h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, d * h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, d, h, w)
            h_ = self.proj_out(h_)
        return x + h_


class ResNetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        dim=2,
        conditioning_dims=None,
        dropout_prob=0.0,
        nca_params={},
        cond_proj_type="zerolinear"
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dim = dim
        assert self.dim in [2, 3], "dim must be 2 or 3"
        self.conditioning_dims = conditioning_dims

        self.nca_params = nca_params
        norm_params = self.nca_params.get("norm_params", {})
        get_act = self.nca_params.get("get_act", lambda: nn.GELU())
        conv_params = self.nca_params.get("conv_params", {})

        self.net1 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_in, **norm_params),
            get_act(),
            get_conv(ch_in, ch_out, dim=self.dim, **conv_params),
        )
        if conditioning_dims is not None:
            self.cond_projs = nn.ModuleList()
            for condition_dim in self.conditioning_dims:
                if cond_proj_type == "zerolinear":
                    self.cond_projs.append(zero_init(nn.Linear(condition_dim, ch_out)))
                elif cond_proj_type == "linear":
                    self.cond_projs.append(nn.Linear(condition_dim, ch_out))
                elif cond_proj_type == "mlp":
                    self.cond_projs.append(
                        nn.Sequential(
                            nn.Linear(condition_dim, ch_out),
                            get_act(),
                            nn.Linear(ch_out, ch_out),
                            get_act(),
                        )
                    )
                else:
                    raise ValueError(f"Unknown cond_proj_type: {cond_proj_type}")
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_out, **norm_params),
            get_act(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            get_conv(ch_out, ch_out, dim=self.dim, init=zero_init, **conv_params),
        )
        if ch_in != ch_out:
            self.skip_conv = get_conv(
                ch_in, ch_out, dim=self.dim, kernel_size=1, padding=0
            )

    def forward(self, x, conditionings=None):
        h = self.net1(x)
        if conditionings is not None:
            assert len(conditionings) == len(self.conditioning_dims)
            assert all(
                [
                    conditionings[i].shape == (x.shape[0], self.conditioning_dims[i])
                    for i in range(len(conditionings))
                ]
            )
            for i, conditioning in enumerate(conditionings):
                conditioning_ = self.cond_projs[i](conditioning)
                if self.dim == 2:
                    h = h + conditioning_[:, :, None, None]
                elif self.dim == 3:
                    h = h + conditioning_[:, :, None, None, None]
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        return x + h


class ResNetDown(nn.Module):
    def __init__(self, resnet_blocks, attention_blocks=None):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.down = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.resnet_blocks[-1].ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x, conditionings, no_down=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if no_down:
            return x, None
        x_skip = x
        x = self.down(x)
        return x, x_skip


class ResNetUp(nn.Module):
    def __init__(
        self, resnet_blocks, attention_blocks=None, ch_out=None, conv_params={}
    ):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.ch_out = ch_out if ch_out is not None else self.resnet_blocks[-1].ch_out
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.up = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
            transposed=True,
        )

    def forward(self, x, x_skip=None, conditionings=None, no_up=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if not no_up:
            x = self.up(x)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return x


# ============= Encoder =============

class Encoder(nn.Module):
    def __init__(
        self,
        shape,
        chs=[48, 96, 192],
        attn_sizes=[],
        mid_attn=False,
        num_res_blocks=1,
        dropout_prob=0.0,
        z_channels=4,
        double_z=True,
        n_attention_heads=1,
        norm_groups=8,
        norm_eps=1e-6,
        norm_affine=True,
        act="gelu",
        conv_kernel_size=3,
        conv_padding_mode="zeros",
    ):
        super().__init__()
        self.shape = shape
        self.in_channels = self.shape[0]
        self.input_size = self.shape[1]
        self.chs = chs
        self.dim = len(self.shape) - 1
        self.attn_sizes = attn_sizes
        self.mid_attn = mid_attn
        if (len(self.attn_sizes) > 0 or self.mid_attn) and self.dim == 3:
            raise ValueError("3D attention very highly discouraged.")
        self.num_res_blocks = num_res_blocks
        self.dropout_prob = dropout_prob
        self.z_channels = z_channels
        self.double_z = double_z
        self.n_attention_heads = n_attention_heads

        assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(num_groups=norm_groups, eps=norm_eps, affine=norm_affine)
        assert act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if act == "gelu":
                return nn.GELU()
            elif act == "relu":
                return nn.ReLU()
            elif act == "silu":
                return nn.SiLU()

        padding = conv_kernel_size // 2
        conv_params = dict(
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=None,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
        )

        self.n_sizes = len(self.chs)
        self.conv_in = get_conv(
            self.in_channels, self.chs[0], dim=self.dim, **conv_params
        )

        curr_size = self.input_size
        self.downs = nn.ModuleList()
        for i_level in range(self.n_sizes):
            ch_in = chs[0] if i_level == 0 else chs[i_level - 1]
            ch_out = chs[i_level]

            resnets = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_out, **resnet_params))
                if curr_size in self.attn_sizes:
                    attentions.append(
                        AttnBlock(
                            ch_out,
                            n_heads=self.n_attention_heads,
                            dim=self.dim,
                            norm_params=norm_params,
                        )
                    )
                ch_in = ch_out
            if len(attentions) == 0:
                attentions = None
            down = ResNetDown(resnets, attentions)
            curr_size = curr_size // 2
            self.downs.append(down)

        # middle
        self.mid1 = ResNetBlock(ch_in, ch_in, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_in,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        self.mid2 = ResNetBlock(ch_in, ch_in, **resnet_params)

        # end
        self.norm_out = nn.GroupNorm(num_channels=ch_in, **norm_params)
        self.act_out = get_act()
        self.conv_out = get_conv(
            in_channels=ch_in,
            out_channels=2 * z_channels if double_z else z_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )

    def forward(self, x):
        # timestep embedding
        conditionings = None

        # downsampling
        h = self.conv_in(x)
        for i, down in enumerate(self.downs):
            h, _ = down(
                h, conditionings=conditionings, no_down=(i == (len(self.downs) - 1))
            )

        # middle
        h = self.mid1(h, conditionings=conditionings)
        if self.mid_attn:
            h = self.mid_attn1(h)
        h = self.mid2(h, conditionings=conditionings)

        # end
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h


# ============= Decoder =============

class Decoder(nn.Module):
    def __init__(
        self,
        shape,
        chs=[48, 96, 192],
        attn_sizes=[],
        mid_attn=False,
        num_res_blocks=1,
        dropout_prob=0.0,
        z_channels=4,
        double_z=True,
        n_attention_heads=1,
        norm_groups=8,
        norm_eps=1e-6,
        norm_affine=True,
        act="gelu",
        conv_kernel_size=3,
        conv_padding_mode="zeros",
    ):
        super().__init__()
        self.shape = shape
        self.in_channels = self.shape[0]
        self.input_size = self.shape[1]
        self.chs = chs
        self.dim = len(self.shape) - 1
        self.attn_sizes = attn_sizes
        self.mid_attn = mid_attn
        if (len(self.attn_sizes) > 0 or self.mid_attn) and self.dim == 3:
            raise ValueError("3D attention very highly discouraged.")
        self.num_res_blocks = num_res_blocks
        self.dropout_prob = dropout_prob
        self.z_channels = z_channels
        self.double_z = double_z
        self.n_attention_heads = n_attention_heads

        assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(num_groups=norm_groups, eps=norm_eps, affine=norm_affine)
        assert act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if act == "gelu":
                return nn.GELU()
            elif act == "relu":
                return nn.ReLU()
            elif act == "silu":
                return nn.SiLU()

        padding = conv_kernel_size // 2
        conv_params = dict(
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=None,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
        )

        self.n_sizes = len(self.chs)

        ch_in = self.chs[-1]
        self.conv_in = get_conv(self.z_channels, ch_in, dim=self.dim, **conv_params)

        self.mid1 = ResNetBlock(ch_in, ch_in, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_in,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        self.mid2 = ResNetBlock(ch_in, ch_in, **resnet_params)

        # upsampling
        curr_size = self.input_size // 2 ** (self.n_sizes - 1)
        self.ups = nn.ModuleList()
        for i_level in reversed(range(self.n_sizes)):
            ch_in = self.chs[i_level]

            resnets = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_in, **resnet_params))
                if curr_size in self.attn_sizes:
                    attentions.append(
                        AttnBlock(
                            ch_in,
                            n_heads=self.n_attention_heads,
                            dim=self.dim,
                            norm_params=norm_params,
                        )
                    )
            if len(attentions) == 0:
                attentions = None
            ch_out = self.chs[0] if i_level == 0 else self.chs[i_level - 1]  # for up
            up = ResNetUp(
                ch_out=ch_out, resnet_blocks=resnets, attention_blocks=attentions
            )
            curr_size = curr_size // 2
            self.ups.append(up)

        self.norm_out = nn.GroupNorm(num_channels=ch_out, **norm_params)
        self.act_out = get_act()
        self.conv_out = get_conv(
            in_channels=ch_out,
            out_channels=self.in_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        conditionings = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid1(h, conditionings=conditionings)
        if self.mid_attn:
            h = self.mid_attn1(h)
        h = self.mid2(h, conditionings=conditionings)

        # upsampling
        for i, up in enumerate(self.ups):
            h = up(h, conditionings=conditionings, no_up=(i == self.n_sizes - 1))

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h


# ============= Main VAE Model =============

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        enc_dec_params,
        embed_dim=8,
        learning_rate=1e-3,
        weight_decay=1.0e-5,
        nll_loss_type="l1",
        kl_weight=0.000001,
        **kwargs,
    ):
        super().__init__()
        self.enc_dec_params = enc_dec_params
        self.encoder = Encoder(**self.enc_dec_params)
        self.decoder = Decoder(**self.enc_dec_params)
        self.dim = self.encoder.dim
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nll_loss_type = nll_loss_type
        assert self.nll_loss_type in [
            "l1",
            "l2",
        ], "nll_loss_type must be l1 or l2"
        self.kl_weight = kl_weight

        z_channels = self.encoder.z_channels
        self.quant_conv = get_conv(
            2 * z_channels, 2 * self.embed_dim, dim=self.dim, kernel_size=1, padding=0
        )
        self.post_quant_conv = get_conv(
            self.embed_dim, z_channels, dim=self.dim, kernel_size=1, padding=0
        )
        # Initialize logvar to a reasonable value
        # log(1000) ≈ 6.9, so exp(logvar) ≈ 1000, which helps with large initial rec_loss
        self.logvar = nn.Parameter(torch.ones(size=(), dtype=torch.float32) * 6.0)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_loss(self, x):
        reconstructions, posteriors = self(x)
        if self.nll_loss_type == "l1":
            rec_loss = nn.functional.l1_loss(x, reconstructions, reduction="none")
        elif self.nll_loss_type == "l2":
            rec_loss = nn.functional.mse_loss(x, reconstructions, reduction="none")
        else:
            raise ValueError("nll_loss_type must be l1 or l2")
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = self.kl_weight * kl_loss
        loss = nll_loss + kl_loss
        metrics = {"kl_loss": kl_loss, "nll_loss": nll_loss, "loss": loss}
        return loss, metrics


# ============= Simple Wrapper =============

class SpectralVAE(nn.Module):
    """Simple wrapper for TEMPO spectral data."""

    def __init__(self, vae):
        super(SpectralVAE, self).__init__()
        self.vae = vae

    def forward(self, x):
        x_rec, _ = self.vae(x)
        return x_rec

    def get_latent(self, x):
        _, latent = self.vae(x)
        return latent

    def get_loss(self, x):
        loss, metrics = self.vae.get_loss(x)
        return loss, metrics

    def get_metrics(self, x):
        _, metrics = self.vae.get_loss(x)
        return metrics


def get_model(model_params, device):
    """Get model exactly like old utils.py"""
    assert model_params["architecture_type"] == "vae"

    # Default encoder/decoder params
    enc_dec_params = dict(
        shape=(1028, 64, 64),
        chs=[512, 256, 128],
        attn_sizes=[],
        mid_attn=True,
        num_res_blocks=1,
        dropout_prob=0.0,
        z_channels=32,
        double_z=True,
        n_attention_heads=4,
        norm_groups=8,
        norm_eps=1e-6,
        norm_affine=True,
        act="gelu",
        conv_kernel_size=3,
        conv_padding_mode="zeros",
    )

    # Update with config params, but filter out non-encoder/decoder params
    config_params = model_params["architecture_params"]["enc_dec_params"]
    for key in enc_dec_params.keys():
        if key in config_params:
            enc_dec_params[key] = config_params[key]

    # Extract VAE-specific params from config
    embed_dim = config_params.get("embed_dim", 32)
    kl_weight = config_params.get("kl_weight", 0.000001)
    nll_loss_type = config_params.get("nll_loss_type", "l1")

    vae = AutoencoderKL(
        enc_dec_params=enc_dec_params,
        embed_dim=embed_dim,
        learning_rate=1e-3,
        weight_decay=1.0e-5,
        nll_loss_type=nll_loss_type,
        kl_weight=kl_weight,
    )
    model = SpectralVAE(vae)
    model = model.to(device)
    assert model_params["optimizer_type"] == "AdamW"
    optimizer = torch.optim.AdamW(model.parameters(), **model_params["optimizer_params"])
    model.optimizer = optimizer
    return model