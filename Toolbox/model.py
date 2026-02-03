import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn
import random

# Set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

class ResidualBlock(nn.Module):
    """
    A residual block with optional normalization,
     downsampling, and learnable skip connection.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        actv (nn.Module): Activation function to
        apply (default: nn.LeakyReLU(0.2)).
        normalize (bool): Whether to apply Instance Normalization (default: False).
        downsample (bool): Whether to downsample the
        spatial resolution by a factor of 2 (default: False).

    Methods:
        forward(x): Forward pass of the residual block.

        Returns:
            torch.Tensor: Output feature map after
            applying both the shortcut and residual paths.

    """

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super().__init__()

        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out

        # Initialize weights in a single step
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)

        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)

        x = self.actv(self.conv1(x))  # Apply activation after first conv
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2)

        if self.normalize:
            x = self.norm2(x)

        x = self.actv(self.conv2(x))  # Apply activation after second conv

        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    """
        Adaptive Instance Normalization (AdaIN) based residual block
        with optional upsampling, downsampling, and learnable skip connection.

        This block applies AdaIN normalization to the residual path,
         enabling style adaptation. It includes options for
        upsampling and downsampling, and incorporates a learnable
         skip connection if the input and output dimensions differ.

        Args:
            dim_in (int): Number of input channels.
            dim_out (int): Number of output channels.
            style_dim (int, optional): Dimensionality of the style vector (default: 64).
            w_hpf (float, optional): Weight for high-pass filtering (default: 0).
            actv (nn.Module, optional): Activation function to apply (default: nn.LeakyReLU(0.2)).
            upsample (bool, optional): Whether to upsample the input (default: False).
            downsample (bool, optional): Whether to downsample the input (default: True).

        Methods:
            forward(x, s): Forward pass of the residual block with style vector `s`.
        Returns:
            torch.Tensor: Output feature map after applying both the shortcut and residual paths.

        """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class Adain_QBlk(nn.Module):
    """
        Adaptive Instance Normalization (AdaIN) layer for style-based transformations.

        This module adjusts the mean and variance of the content input `x` according to
        the style vector `s`, allowing for style adaptation.

        Args:
            style_dim (int): Dimensionality of the style vector.
            num_features (int): Number of feature channels in the content input.

        Methods:
            forward(x, s): Applies AdaIN transformation to input `x` using style vector `s`.
        Returns:
            torch.Tensor: Output feature map after AdaIN transformation.
        """

    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False, downsample=True):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out

        # Build weights in a single step
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        if self.learned_sc:
            x = self.conv1x1(x)

        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2)

        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.downsample:
            x = F.avg_pool2d(x, 2)

        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)

        return x

    def forward(self, x, s):
        out = self._residual(x, s)

        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)

        return out


class HighPass(nn.Module):
    """
    High-pass filter module for emphasizing high-frequency components in the input image.

    This module applies a high-pass filter to the input image to enhance edges and fine details by
    subtracting low-frequency components. The filter is applied using a convolution operation.

    Args:
        w_hpf (float): Weight for high-pass filtering. This scales the filter to adjust its impact.
        device (torch.device): Device on which the filter tensor should be registered.

    Methods:
        forward(x): Applies the high-pass filter to the input tensor `x` using convolution.

        Returns:
            torch.Tensor: Output tensor after applying the high-pass filter.
    """

    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        # print(x.size(1), filter.shape)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)

class Generator_Rx(nn.Module):
    """
       Generator network for image synthesis with
       adaptive instance normalization (AdaIN) and high-pass filtering.

       This generator network creates images by progressively
       upsampling feature maps, incorporating style information
       through AdaIN blocks. It supports high-pass filtering to enhance image details.

       Args:
           img_size (int, optional): Size of the input image (default: 128).
           style_dim (int, optional): Dimensionality of the style vector (default: 64).
           max_conv_dim (int, optional): Maximum number of convolutional channels (default: 512).
           w_hpf (float, optional): Weight for high-pass filtering (default: 1).

       Methods:
           forward(x, s, masks=None): Forward pass of the generator,
            producing feature maps and optionally applying masks.

       Returns:
            tuple: Tuple containing feature maps at different resolutions.
                - x_64: Feature map at 64x64 resolution.
                - x_128: Feature map at 128x128 resolution.
                - x_256: Feature map at 256x256 resolution.
                - x_512: Feature map at 512x512 resolution.
       """
    def __init__(self, img_size=128, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size

        # Create sub-layer pipelines for downscaled analysis
        self.layers_1 = nn.ModuleList()
        self.layers_enc = nn.ModuleList()
        self.layers_dec = nn.ModuleList()
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.layers_1 = nn.Conv2d(3, dim_in, 3, 1, 1)
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1

        ## Encoder/Decoder Block ##
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.layers_enc.append(Adain_QBlk(dim_in, dim_out, style_dim,
                                              w_hpf=w_hpf, actv=nn.LeakyReLU(0.2),
                                              upsample=False, downsample=True))
            dim_in = dim_out

        for idx in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.layers_dec.insert(0, Adain_QBlk(dim_in, dim_out, style_dim,
                                                 w_hpf=w_hpf, actv=nn.LeakyReLU(0.2),
                                                 upsample=True, downsample=False))
            dim_in = dim_out

        ## FC Block ##
        for _ in range(2):
            self.layers_enc.append(Adain_QBlk(dim_in, dim_out, style_dim,
                                              w_hpf=w_hpf, actv=nn.LeakyReLU(0.2),
                                              upsample=False, downsample=False))

        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, 3, 1, 1, 0))

        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

    def forward(self, x, s, masks=None):
        x = self.layers_1(x)
        feats = []
        cache = {}
        for idx, block in enumerate(self.layers_enc):
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x, s)
            feats.append(x)

        x = self.to_rgb(x)
        while len(feats) < 4:
            feats.append(None)
        x_64, x_128, x_256, x_512 = feats[:4]
        return x, x_64, x_128, x_256, x_512


class Discriminator(nn.Module):
    """
        Discriminator network for distinguishing between real and generated images,
        and classifying images into domains.

        This network processes images to produce a score indicating their
        authenticity and classify them into different domains.

        Args:
            img_size (int, optional): Size of the input image (default: 128).
            num_domains (int, optional): Number of domains for classification (default: 2).
            max_conv_dim (int, optional): Maximum number of convolutional channels (default: 512).

        Methods:
            forward(x, y): Forward pass through the discriminator, producing
            classification scores for the input images.

        Returns:
            torch.Tensor: Classification scores for
            the input images, shape (batch_size).

        """
    def __init__(self, img_size=128, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResidualBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


class StyleEncoder(nn.Module):
    """
    Style encoder network for extracting style features from images.
    This network extracts style features from
    images and maps them to a style vector for each domain.

    Args:
        img_size (int, optional): Size of the input image (default: 128).
        style_dim (int, optional): Dimensionality of the style vector (default: 64).
        num_domains (int, optional): Number of domains for
         style vector extraction (default: 2).
        max_conv_dim (int, optional): Maximum number
         of convolutional channels (default: 512).

    Methods:
        forward(x, y): Forward pass through the encoder,
         producing style vectors for the input images.

    Returns:
            torch.Tensor: Style vectors for the
            input images, shape (batch_size, style_dim).
    """

    def __init__(self, img_size=128, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResidualBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []

        for layer in self.unshared:
            out += [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)

        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)

        return s
