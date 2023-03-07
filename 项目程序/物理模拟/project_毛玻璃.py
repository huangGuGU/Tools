from 装饰器.decorator_程序启动 import logit
import os

import cv2
import PIL.Image as Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import torch.nn
from torch import pi


class Diffraction(torch.nn.Module):
    def __init__(self, lam, size):
        super(Diffraction, self).__init__()
        # optical basic variable
        # light
        self.lam = lam
        self.k = 2 * pi / self.lam
        # diffraction fringe
        self.pixel_size = self.lam

        # model
        self.size = size.clone().detach()

        # k-space coordinate
        self.u = torch.fft.fftshift(torch.fft.fftfreq(self.size[0], self.pixel_size))
        self.v = torch.fft.fftshift(torch.fft.fftfreq(self.size[1], self.pixel_size))
        self.fu, self.fv = torch.meshgrid(self.u, self.v, indexing='xy')

        # frequency response function
        self.h = lambda fu, fv, wl, z: \
            torch.exp((1.0j * 2 * pi * z / wl) * torch.sqrt(1 - (wl * fu) ** 2 - (wl * fv) ** 2))

        # frequency limit (book is sqrt(fu^2 + fv^2) < 1/lam)
        self.limit = 1 / self.lam

    def light_forward(self, images, distance):
        k_images = torch.fft.fft2(images)
        k_images = torch.fft.fftshift(k_images, dim=(2, 3))

        mask_input = torch.hypot(self.fu, self.fv) < self.limit
        h_input_limit = (mask_input * self.h(self.fu, self.fv, self.lam, distance * self.lam))
        k_output = k_images * h_input_limit

        output = torch.fft.ifft2(k_output)
        return output

    # ################################################################################################################################################################################################################################


'''毛玻璃的仿真'''


class Diffuser(torch.nn.Module):

    def __init__(self, n0, n1, lam, size):
        super(Diffuser, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.delta_n = self.n1 - self.n0
        self.lam = lam
        self.size = torch.tensor(size)

    def diffuserplane(self, miu, sigma0, sigma):

        kernel = self.gaussian_2d_kernel(
            torch.tensor((torch.sqrt(2 * torch.log(torch.tensor(2))) * sigma) / self.lam,
                         dtype=torch.int16) * 2 + 1, sigma).unsqueeze(0).unsqueeze(0)

        diffuser = torch.FloatTensor(torch.normal(miu, sigma0,
                                                  size=(self.size[0] + kernel.shape[2] - 1,
                                                        self.size[1] + kernel.shape[3] - 1))).unsqueeze(0).unsqueeze(0)

        # import matplotlib.pyplot as plt
        #
        # def function_R(x, y):
        #     return torch.exp(-torch.pi * (x ** 2 + y ** 2) / 3.2 * self.lam)

        # x = diffuser.squeeze(0).squeeze(0)[-2]
        # y = diffuser.squeeze(0).squeeze(0)[-1]
        # X, Y = torch.meshgrid(x, y)
        # R = function_R(X, Y)
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(X, Y, R, rstride=1, cstride=1,
        #                 cmap='viridis', edgecolor='none')
        # plt.show()

        return torch.conv2d(diffuser, kernel, padding=0).squeeze(0).squeeze(0)

    def gaussian_2d_kernel(self, kernel_size, Sigma):
        kernel = torch.zeros([kernel_size, kernel_size])
        center = kernel_size // torch.tensor(2)

        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        y = torch.linspace(kernel_size // 2, -kernel_size // 2, kernel_size)
        mask_x, mask_y = torch.meshgrid(x, y, indexing='xy')
        mask_rho = torch.hypot(mask_x, mask_y)
        mask = (mask_rho < center)

        if Sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

        s = 2 * (Sigma ** 2)
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                x = (i - center) * self.lam
                y = (j - center) * self.lam

                kernel[i, j] = torch.exp(torch.div(-(x ** 2 + y ** 2), s))

        kernel = kernel * mask
        kernel = kernel / torch.sum(kernel)

        return torch.FloatTensor(kernel)


class incoherent_light_Net(torch.nn.Module):
    def __init__(self, num_layers, size, lam, ratio,
                 miu, sigma0, sigma, n0, n1):
        # 毛玻璃的参数   220目：miu=63*10**-6,sigma0=14*10**-6,sigma=15.75*10**-6
        #             600目：miu=16*10**-6,sigma0=5*10**-6,sigma=4*10**-6
        super(incoherent_light_Net, self).__init__()
        self.num_layers = num_layers

        self.size = torch.tensor(size)
        self.lam = lam

        self.n0 = n0
        self.n1 = n1
        self.miu = miu
        self.sigma0 = sigma0
        self.sigma = sigma
        self.dropout = torch.nn.Dropout(ratio)
        self.random_diffuser1 = 2 * pi * torch.rand(self.size[0], self.size[1], device='cpu')
        self.random_diffuser2 = 2 * pi * torch.rand(self.size[0], self.size[1], device='cpu')

        '''毛玻璃的参数'''
        D = Diffuser(lam=self.lam, n0=self.n0, n1=self.n1,
                     size=self.size).diffuserplane(miu=self.miu, sigma0=self.sigma0,
                                                   sigma=self.sigma)
        phase = (1.5 - 1) * D / self.lam
        phase = 2 * torch.pi * (phase - phase.min()) / (phase.max() - phase.min())
        self.register_buffer(f'diffuser', phase)

        # learning weight
        self.phase_modulator = []
        for layer in range(self.num_layers):
            self.phase_modulator.append(
                torch.nn.Parameter(torch.rand(size=(self.size[0], self.size[1]))))
            self.register_parameter(f'phase_modulator_{layer}', self.phase_modulator[layer])

        self.phase_output = torch.nn.Parameter(2 * pi * torch.rand(size=(self.size[0], self.size[1])))

    def forward(self, inputs):

        diffraction = Diffraction(lam=self.lam, size=self.size)

        # light = torch.ones(inputs.shape, device='cpu')
        # light_incoherent = light * torch.exp(1.0j * self.random_diffuser1)
        # light_incoherent = diffraction.light_forward(light_incoherent, 53)
        # inputs = inputs * light_incoherent

        x = diffraction.light_forward(inputs, 53)


        # M = D.max()
        # m = D.min()
        # c = M - m
        # print(c)
        # plt.imshow((D.cpu().numpy()))
        # plt.colorbar()
        # plt.title('D')
        # plt.show()


        plt.imshow((self.diffuser.cpu().numpy()))
        plt.axis('off')
        plt.colorbar()
        plt.title('phase')
        plt.show()
        x = x * torch.exp(1.0j * self.diffuser)

        # ################################################################################################################################################################################################################################

        '''调制层'''

        for index, phase in enumerate(self.phase_modulator):
            x = diffraction.light_forward(x, 2.7)

            # One = torch.ones((self.size[0], self.size[1]), device='cpu')
            # dropout_mask = self.dropout(One)
            # x = x * torch.exp(1.0j * 2 * pi * torch.sigmoid(phase)) * dropout_mask
            # print(f'phase_modulator_{index}:{phase}')

        # ################################################################################################################################################################################################################################

        '''输出层'''
        x = diffraction.light_forward(x, 9.3)
        x = torch.abs(x) ** 2
        x = torch.tanh(x)  # 1
        plt.imshow((x[0,0,:,:].cpu().numpy()))
        plt.colorbar()
        plt.title('D')
        plt.show()

        return x


if __name__ == '__main__':
    # 120
    # 目
    # miu = 100 * 10 ** -6, sigma0 = 20 * 10 ** -6, sigma = 20 * 10 ** -6
    #
    # 220
    # 目
    # miu = 63 * 10 ** -6, sigma0 = 14 * 10 ** -6, sigma = 15.75 * 10 ** -6
    #
    # 600
    # 目
    # miu = 16 * 10 ** -6, sigma0 = 5 * 10 ** -6, sigma = 4 * 10 ** -6
    #
    # 1500
    # 目
    # miu = 10 * 10 ** -6, sigma0 = 2 * 10 ** -6, sigma = 2 * 10 ** -6

    x1 = torch.rand([4, 1, 240, 240])
    label1 = torch.rand([4, 1, 240, 240])
    models = incoherent_light_Net(num_layers=4, size=(240, 240), lam=632e-9,
                                  miu = 63 * 10 ** -6, sigma0 = 14 * 10 ** -6, sigma = 15.75 * 10 ** -6, n0=1,
                                  n1=1.52, ratio=0.3)

    # for xy in models.state_dict():
    #     print(xy, models.state_dict()[xy])
    # print('*' * 50)
    # for xy in models.parameters():
    #     print(xy)
    # print('*' * 50)

    loss_function = torch.nn.MSELoss()

    prediction1 = models(x1)
    m1 = torch.mean(prediction1)
    s1 = torch.std(prediction1)
    loss = loss_function(prediction1, label1)
    print(loss)

    pass
