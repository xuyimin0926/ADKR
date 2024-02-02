import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


""" class Mask_s(nn.Module):
    '''
        Attention Mask spatial.
    '''

    def __init__(self, planes, eps=0.66667, bias=-1, **kwargs):
        super(Mask_s, self).__init__()
        # Parameter
        #self.width, self.height, self.channel = w, h, planes
        #self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
        # self.eleNum_s = torch.Tensor([self.mask_h * self.mask_w])
        # spatial attention
        self.atten_s = nn.Conv2d(planes, 1, kernel_size=3, stride=1, bias=bias >= 0, padding=1)
        #initialize
        torch.nn.init.xavier_uniform(self.atten_s.weight)
        if bias >= 0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(eps=eps)
        # Norm
        # self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x):
        batch, channel, height, width = x.size()
        # Pooling
        #input_ds = F.adaptive_avg_pool2d(input=x, output_size=(self.mask_h, self.mask_w))
        # spatial attention
        s_in = self.atten_s(x)  # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in)  # [N, 1, h, w]
        # norm
        # norm = self.norm(mask_s)
        # norm_t = self.eleNum_s.to(x.device)
        return mask_s """
class Mask_s(nn.Module):
    '''
        Attention Mask spatial.
    '''

    def __init__(self, planes, eps=0.66667, bias=-1, **kwargs):
        super(Mask_s, self).__init__()
        # Parameter
        # self.width, self.height, self.channel = w, h, planes
        # self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
        # self.eleNum_s = torch.Tensor([self.mask_h * self.mask_w])
        # spatial attention
        self.atten_s = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, bias=bias >= 0, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, bias=bias >= 0, padding=1)
        )
        # initialize
        self.atten_s.apply(self.weight_init)
        if bias >= 0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(eps=eps)
        # Norm
        # self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x):
        batch, channel, height, width = x.size()
        # Pooling
        # input_ds = F.adaptive_avg_pool2d(input=x, output_size=(self.mask_h, self.mask_w))
        # spatial attention
        s_in = self.atten_s(x.mean(dim=1, keepdim=True))  # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in)  # [N, 1, h, w]
        # norm
        # norm = self.norm(mask_s)
        # norm_t = self.eleNum_s.to(x.device)
        return mask_s

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight)

class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''

    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor + eps) - torch.log(
            1 - uniform_samples_tensor + eps)
        return gumble_samples_tensor

    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits

    def forward(self, logits):
        if not self.training:
            out_hard = (logits >= 0).float()
            return out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard


class GumbelSigmoid(nn.Module):
    def __init__(self, max_T, decay_alpha):
        super(GumbelSigmoid, self).__init__()

        self.max_T = max_T
        self.decay_alpha = decay_alpha
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

        self.register_buffer('cur_T', torch.tensor(max_T))

    def forward(self, x):
        if self.training:
            _cur_T = self.cur_T
        else:
            _cur_T = 0.03

        # Shape <x> : [N, C, H, W]
        # Shape <r> : [N, C, H, W]
        r = 1 - x
        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        # Generate Noise
        x_N = torch.rand_like(x)
        r_N = torch.rand_like(r)
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        # Get Final Distribution
        x = x + x_N
        x = x / (_cur_T + self.p_value)
        r = r + r_N
        r = r / (_cur_T + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)
        x = x[:, [0], :, :]

        if self.training:
            self.cur_T = self.cur_T * self.decay_alpha

        return x
    

class GaussianKernel(nn.Module):
    def __init__(self, size):
        super(GaussianKernel, self).__init__()

        s = (size - 1) // 2
        _x = torch.linspace(-s, s, size).reshape((size, 1)).repeat((1, size))
        _y = torch.linspace(-s, s, size).reshape((1, size)).repeat((size, 1))
        self.d = _x ** 2 + _y ** 2

    def forward(self, sigma):
        k = sigma ** 2
        A = k / (2. * np.pi)
        d = -k / 2. * self.d.cuda()
        B = torch.exp(d)
        B = A * B
        return B
   