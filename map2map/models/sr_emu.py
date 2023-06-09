from math import log2
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from .narrow import narrow_by
from .resample import Resampler
from .style import ConvStyled3d

class AddNoise(nn.Module):
    """Add custom noise.

    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, chan):
        super().__init__()

        self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x, noise):
        assert x.shape[1] == self.std.shape[0] == noise.shape[1]
        std = self.std.view(1, -1, 1, 1, 1)
        x = x + std * noise
        return x

class LeakyReLUStyled(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super().__init__(negative_slope, inplace)

    """ Trivially evaluates standard leaky ReLU, but accepts second argument

    for style array that is not used
    """

    def forward(self, x, style=None):
        return super().forward(x)
    
class BatchNormStyled3d(nn.BatchNorm3d):
    """ Trivially does standard batch normalization, but accepts second argument

    for style array that is not used
    """

    def forward(self, x, style=None):
        return super().forward(x)


class Generator(nn.Module):
    def __init__(self,
                 in_chan,
                 out_chan,
                 style_size,
                 scale_factor=8,
                 chan_base=512,
                 chan_min=64,
                 chan_max=512,
                 **kwargs
                 ):
        """ The StyleGAN2 generator.

        Args:
            in_chan (torch.Tensor): input channel, on default 3+3 for displacement and velocity
            out_chan (torch.Tensor): output channel, on default 3+3 for displacement and velocity
            style_size (_type_): dimension of the style vector, on default (1,1)
            scale_factor (int, optional): upscaling factor. Defaults to 8.
            chan_base (int, optional): base channel number. Defaults to 512.
            chan_min (int, optional): minimum channel number. Defaults to 64.
            chan_max (int, optional): maximum channel number. Defaults to 512.

        Returns:
            y: super-resolution output
        """
        super().__init__()

        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))
        self.num_blocks = num_blocks

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c
        

        self.convblock0 = ConvStyled3d(in_chan, chan(0), self.style_size, 1)
        self.act = LeakyReLUStyled(0.2, True)
        
        self.addnoise = AddNoise(chan=chan(0))
        
        hblock_channels = [[chan(i), chan(i+1)] for i in range(num_blocks)]
        
        hblock = partial(HBlock, in_chan=in_chan, out_chan=out_chan, style_size=style_size)
        
        self.hblock0 = hblock(*hblock_channels[0])
        self.hblock1 = hblock(*hblock_channels[1])
        self.hblock2 = hblock(*hblock_channels[2])
        
        self.noise_proj = ConvStyled3d(6, chan(0), self.style_size, 1)
        

    def forward(self, 
                x: torch.Tensor, 
                style: torch.Tensor, 
                early_noise: torch.Tensor,
                noise0: torch.Tensor,
                noise1: torch.Tensor,
                noise2: torch.Tensor,
                ):
        y = x  # direct upsampling from the input
        
        noise01 = noise0
        noise02 = narrow_by(noise0, 1)
        del noise0
        
        noise11 = noise1
        noise12 = narrow_by(noise1, 1)
        del noise1
        
        noise21 = noise2
        noise22 = narrow_by(noise2, 1)
        del noise2
        
        x = self.convblock0((x, style))
        x = self.act(x)
        
        # NOTE this should exist (at least it's in original StyleGAN2 code)
        # 1st noise
        early_noise = self.noise_proj((early_noise, style))
        x = self.addnoise(x, early_noise)
        
        
        noise0 = [noise01, noise02]
        noise1 = [noise11, noise12]
        noise2 = [noise21, noise22]
        # start HBlocks
        
        x, y = self.hblock0(x, y, style, noise0)
        x, y = self.hblock1(x, y, style, noise1)
        x, y = self.hblock2(x, y, style, noise2)
        
        
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

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(self, prev_chan, next_chan, in_chan, out_chan, style_size):
        super().__init__()
        
        self.act = LeakyReLUStyled(0.2, True)
        self.upsample = Resampler(3, 2)
        
        self.conv1 = ConvStyled3d(prev_chan, next_chan, style_size, 3)
        self.addnoise1 = AddNoise(chan=next_chan)

        
        self.conv2 = ConvStyled3d(next_chan, next_chan, style_size, 3)
        self.addnoise2 = AddNoise(chan=next_chan) 
        
        
        self.proj = ConvStyled3d(next_chan, out_chan, style_size, 1)
        
        self.noise_proj1 = ConvStyled3d(in_chan, next_chan, style_size, 1)
        self.noise_proj2 = ConvStyled3d(in_chan, next_chan, style_size, 1)

    def forward(self, x, y, s, noise):
        # ------------ Left branch ------------
        # 16 pad 3 = 22
        x = self.upsample(x)
        # 44 - 2 = 42
        
        # first styled block
        x = self.conv1((x,s))
        x = self.act(x)
        # 42 - 2 = 40 = 32 pad 4
        
        # noise adding part
        noise0 = self.noise_proj1((noise[0], s))
        x = self.addnoise1(x, noise0)
        
    
        # second styled block
        x = self.conv2((x,s))
        x = self.act(x)
        # 40 - 2 = 38 = 32 pad 3
        
        # noise adding part
        noise1 = self.noise_proj2((noise[1], s))
        x = self.addnoise2(x, noise1)
          
        # ------------ Right branch(RGB branch) ------------
        y = self.upsample(y)
        # 44 - 2 = 42 
        
        # crop y to match x
        x_size = x.shape[-1]
        y_size = y.shape[-1]
        if x_size != y_size:
            narrow_edge = (y_size - x_size) // 2
            y = narrow_by(y, narrow_edge)
            # 42 - 4 = 38 = 32 pad 3
            
        # To RGB block
        feature_map = self.proj((x,s))
        y = y + self.act(feature_map)
        
        return x, y
    
    
