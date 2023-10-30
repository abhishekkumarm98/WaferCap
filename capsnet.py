import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    


class WaferCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(WaferCaps, self).__init__()
        
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Encoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1, wc_out_channels=16, numCls = 9, num_capsules = 10):
        super(Encoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.wc_out_channels = wc_out_channels
        self.fully_layers = nn.Sequential(nn.Linear(self.wc_out_channels*num_capsules, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(512, numCls), nn.Sigmoid())
            
    
    def forward(self, x):
        return self.fully_layers(x.view(-1, self.wc_out_channels*10))


class CapsNet(nn.Module):
    def __init__(self, config=None, percentSeenCls = 100):
        super(CapsNet, self).__init__()
        
        numCls = int(np.round((percentSeenCls/100) * 9))
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels, config.pc_kernel_size, config.pc_num_routes)
            self.wafer_capsules = WaferCaps(config.wc_num_capsules, config.wc_num_routes, config.wc_in_channels, config.wc_out_channels)
            self.encoder = Encoder(config.input_width, config.input_height, config.cnn_in_channels, config.wc_out_channels, numCls, config.wc_num_capsules)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.wafer_capsules = WaferCaps()
            self.encoder = Encoder()
        
        self.cat_loss = nn.BCELoss()

    
    def forward(self, data):
        output = self.wafer_capsules(self.primary_capsules(self.conv_layer(data)))
        return self.encoder(output)

    def loss(self, data, target):
        return self.cat_loss(data, target)
        