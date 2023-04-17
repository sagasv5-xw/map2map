from math import log2
import numpy as np
import torch
import torch.nn as nn

from .narrow import narrow_by
from .resample import Resampler, Resampler2
from .style import ConvStyled3d, LeakyReLUStyled, LeakyReLUStyled2
from .styled_conv import ResStyledBlock
from .lag2eul import lag2eul


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class G(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512, cat_noise=False,
                 **kwargs):
        super().__init__()

        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(0), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            self.blocks.append(
                HBlock(prev_chan, next_chan, out_chan, cat_noise, style_size))

    def forward(self, x, style):
        s = style
        y = x  # direct upsampling from the input

        x = self.block0((x, s))

        # y = None  # no direct upsampling from the input

        for block in self.blocks:
            x, y, s = block(x, y, s)
        return y


class HBlock(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(self, prev_chan, next_chan, out_chan, cat_noise, style_size):
        super().__init__()

        self.upsample = Resampler(3, 2)

        # isolate conv style part to make input right
        self.noise_upsample = nn.Sequential(
            AddNoise(cat_noise, chan=prev_chan),
            self.upsample,
        )

        self.conv = nn.Sequential(
            ConvStyled3d(prev_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )
        self.addnoise = AddNoise(cat_noise, chan=next_chan) 
        
        self.conv1 = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )

        self.proj = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), out_chan, style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

    def forward(self, x, y, s):
        x = self.noise_upsample(x)
        x = self.conv((x,s))
        x = self.addnoise(x)
        x = self.conv1((x,s))

        if y is None:
            y = self.proj((x,s))
        else:
            y = self.upsample(y)  

            y = narrow_by(y, 2)
            y = y + self.proj((x,s))
        return x, y, s


class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, chan=1):
        super().__init__()

        self.cat = cat

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1])

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x


class D(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=8,
                 chan_base=512, chan_min=64, chan_max=512,
                 **kwargs):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))
        self.style_size = style_size

        assert chan_min <= chan_max

        def chan(b):
            if b >= 0:
                c = chan_base >> b
            else:
                c = chan_base << -b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan + 8, chan(num_blocks), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )
        # FIXME here I hard coded the in_chan+8 to meet the dimension after mesh_up factor 2

        self.blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            prev_chan, next_chan = chan(b + 1), chan(b)
            self.blocks.append(ResStyledBlock(in_chan=prev_chan, out_chan=next_chan, style_size=style_size, seq='CACA',
                                                last_act=False))
            self.blocks.append(Resampler2(3, 0.5))

        self.block9 = nn.Sequential(
            ConvStyled3d(chan(0), chan(-1), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )
        self.block10 = ConvStyled3d(chan(-1), 1, self.style_size, 1)

    def forward(self, x, style):
        # lag to eul
        s = style
        # lag_x = x[:, :3]
        # rs = np.float(s)
        # eul_x = lag2eul(lag_x, a=rs)[0]
        # x = torch.cat([eul_x, x], dim=1)
        
        # D start
        x = self.block0((x, s))

        for block in self.blocks:
            x = block((x, s))

        x = self.block9((x, s))
        x = self.block10((x, s))

        return x
