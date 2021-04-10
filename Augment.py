# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:45:16 2021
@author: Jason
Discriminator Augment Pipeline
"""
'''import'''
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.utils
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import upfirdn2d
from torch_utils.ops import grid_sample_gradfix
from torch_utils.ops import conv2d_gradfix
import warnings
warnings.filterwarnings('ignore')
'''parameters of filters'''
wavelets = {
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
}
'''functions to construct transformation matrix'''
def matrix(*rows):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows))
    elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))
# translation
def translate2d(tx, ty):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    )
def translate2d_inv(tx, ty):
    return translate2d(-tx, -ty)
# scale
def scale2d(sx, sy):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
    )
def scale2d_inv(sx, sy):
    return scale2d(1 / sx, 1 / sy)
# rotation
def rotate2d(theta):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
    )
def rotate2d_inv(theta):
    return rotate2d(-theta)
'''Augment Pipeline'''
class AugmentPipe(nn.Module):
    def __init__(self, 
        xflip=0.2, rotate90=0.2, xint=0.2, xint_max=0.125,
        imgfilter=0.2, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0.3, cutout=0.6, noise_std=0.1, cutout_size=0.5,              
    ):
        super(AugmentPipe, self).__init__()
        self.p = 0.4
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.
        # Image-space filtering.
        self.imgfilter        = float(imgfilter)        # Probability multiplier for image-space filtering.
        self.imgfilter_bands  = list(imgfilter_bands)   # Probability multipliers for individual frequency bands.
        self.imgfilter_std    = float(imgfilter_std)    # Log2 standard deviation of image-space filter amplification.
        # Image-space corruptions.
        self.noise            = float(noise)            # Probability multiplier for additive RGB noise.
        self.cutout           = float(cutout)           # Probability multiplier for cutout.
        self.noise_std        = float(noise_std)        # Standard deviation of additive RGB noise.
        self.cutout_size      = float(cutout_size)      # Size of the cutout rectangle, relative to image dimensions.
        self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6']))
        Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))
    def forward(self, image):
        batch_size, channel, height, width = image.shape
        device = image.device
        I_3 = torch.eye(3, device=device)
        G_inv = I_3
        # apply x-flip
        i = torch.floor(torch.rand([batch_size], device=device) * 2)
        i = torch.where(torch.rand([batch_size], device=device) < self.xflip * self.p, i, torch.zeros_like(i))
        G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)
        i = torch.floor(torch.rand([batch_size], device=device) * 4)
        i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
        G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)
        t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
        t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
        G_inv = G_inv @ translate2d_inv(torch.round(t[:,0] * width), torch.round(t[:,1] * height))
        cx = (width - 1) / 2
        cy = (height - 1) / 2
        cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]).to(device) # [idx, xyz]
        cp = G_inv @ cp.t() # [batch, xyz, idx]
        Hz_pad = self.Hz_geom.shape[0] // 4
        margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
        margin = torch.cat([-margin, margin]).max(dim=1).values # [x0, y0, x1, y1]
        margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
        margin = margin.max(misc.constant([0, 0] * 2, device=device))
        margin = margin.min(misc.constant([width-1, height-1] * 2, device=device))
        mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)
        # Pad image and adjust origin.   1*32*32 --- 1*51*51
        image = torch.nn.functional.pad(input=image, pad=[mx0,mx1,my0,my1], mode='reflect')
        G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv
        # Upsample.
        image = upfirdn2d.upsample2d(x=image, f=self.Hz_geom, up=2)
        G_inv = scale2d(2, 2).to(device) @ G_inv @ scale2d_inv(2, 2).to(device)
        G_inv = translate2d(-0.5, -0.5).to(device) @ G_inv @ translate2d_inv(-0.5, -0.5).to(device)
        # Execute transformation.
        shape = [batch_size, channel, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
        G_inv = scale2d(2 / image.shape[3], 2 / image.shape[2]).to(device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2]).to(device)
        grid = torch.nn.functional.affine_grid(theta=G_inv[:,:2,:], size=shape, align_corners=False)
        image = grid_sample_gradfix.grid_sample(image, grid)
        # Downsample and crop.
        image = upfirdn2d.downsample2d(x=image, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)
        num_bands = self.Hz_fbank.shape[0]
        assert len(self.imgfilter_bands) == num_bands
        expected_power = misc.constant(np.array([10, 1, 1, 1]) / 13, device=device) # Expected power spectrum (1/f).

        # Apply amplification for each band with probability (imgfilter * strength * band_strength).
        g = torch.ones([batch_size, num_bands], device=device) # Global gain vector (identity).
        for i, band_strength in enumerate(self.imgfilter_bands):
            t_i = torch.exp2(torch.randn([batch_size], device=device) * self.imgfilter_std)
            t_i = torch.where(torch.rand([batch_size], device=device) < self.imgfilter * self.p * band_strength, t_i, torch.ones_like(t_i))
            t = torch.ones([batch_size, num_bands], device=device)                  # Temporary gain vector.
            t[:, i] = t_i                                                           # Replace i'th element.
            t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt() # Normalize power.
            g = g * t                                                               # Accumulate into global gain.

        # Construct combined amplification filter.
        Hz_prime = g @ self.Hz_fbank                                    # [batch, tap]
        Hz_prime = Hz_prime.unsqueeze(1).repeat([1, channel, 1])   # [batch, channels, tap]
        Hz_prime = Hz_prime.reshape([batch_size * channel, 1, -1]) # [batch * channels, 1, tap]

        # Apply filter.
        p = self.Hz_fbank.shape[1] // 2
        image = image.reshape([1, batch_size * channel, height, width])
        image = torch.nn.functional.pad(input=image, pad=[p,p,p,p], mode='reflect')
        image = conv2d_gradfix.conv2d(input=image, weight=Hz_prime.unsqueeze(2), groups=batch_size*channel)
        image = conv2d_gradfix.conv2d(input=image, weight=Hz_prime.unsqueeze(3), groups=batch_size*channel)
        image = image.reshape([batch_size, channel, height, width])
        sigma = torch.randn([batch_size, 1, 1, 1], device=device).abs() * self.noise_std
        sigma = torch.where(torch.rand([batch_size, 1, 1, 1], device=device) < self.noise * self.p, sigma, torch.zeros_like(sigma))
        image = image + torch.randn([batch_size, channel, height, width], device=device) * sigma
        size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=device)
        size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=device) < self.cutout * self.p, size, torch.zeros_like(size))
        center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
        coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
        coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
        mask_x = (((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2)
        mask_y = (((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2)
        mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
        image = image * mask
        return image