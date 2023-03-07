import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import PIL.Image as Image


size = 200
lam = 689e-9
# lam = 0.00072
D = size * lam

distance = 900*lam
pi = torch.pi
scale = 1


class Diffraction(torch.nn.Module):

    def __init__(self):
        super(Diffraction, self).__init__()
        """optical basic variable"""
        # light
        self.lam = lam
        self.k = 2 * pi / self.lam

        # diffraction fringe
        self.scale = scale
        self.distance = distance
        self.pixel_size = self.lam / self.scale
        self.size = size

        """model related setting"""
        # space variable

        self.z = self.distance


        # k-space coordinate
        self.u = torch.fft.fftshift(torch.fft.fftfreq(self.size, self.pixel_size))
        self.v = torch.fft.fftshift(torch.fft.fftfreq(self.size, self.pixel_size))
        self.fu, self.fv = torch.meshgrid(self.u, self.v, indexing='xy')

        # frequency response function (book is H(fx,fy)=exp{jkz*sqrt[1-(λfx)^2-(λfY)^2]})
        self.h = lambda fu, fv, wl, z: \
            torch.exp((1.0j * 2 * pi * z / wl) * torch.sqrt(1 - (wl * fu) ** 2 - (wl * fv) ** 2))
        # frequency limit (book is sqrt(fu^2 + fv^2) < 1/lam)
        self.limit = 1 / self.lam



    def light_input_forward(self, images):
        k_images = torch.fft.fft2(images)
        k_images = torch.fft.fftshift(k_images, dim=(2, 3))

        mask_input = torch.hypot(self.fu, self.fv) < self.limit

        h = self.h(self.fu, self.fv, self.lam, self.z)
        h[torch.isnan(h)] = 0

        h_input_limit = (mask_input * h)
        k_output = k_images * h_input_limit
        output = torch.fft.ifft2(k_output)

        return output

    @staticmethod
    def out_img(img_out):
        # 图片分辨率 = figsize*dpi 代码为512*512
        # plt.rcParams['figure.figsize'] = (10.24, 10.24)
        # plt.rcParams['savefig.dpi'] = 300
        # 去除白框
        plt.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # 保存图片，cmap为调整配色方案
        plt.imshow(img_out[0, :, :], cmap=plt.cm.gray, interpolation='gaussian')
        # plt.savefig(target_path)
        # plt.savefig(path,dpi = 300,format = 'png',pad_inches = 0,transparent = True)

        plt.show()


path = '/Users/hzh/Desktop/x'
target = '/Users/hzh/Desktop/y'
if os.path.exists(target):
    pass
else:
    os.mkdir(target)

img_list = os.listdir(path)
try:
    img_list.remove('.DS_Store')
except:
    pass

for img in img_list:
    img_path = os.path.join(path, img)
    target_path = os.path.join(target, img)

    img = Image.open(img_path).convert('L')





    img = transforms.Resize([size,size])(img)
    img_in = transforms.ToTensor()(img)


    diffraction = Diffraction()
    img_in = torch.unsqueeze(img_in, 0)
    phase = 1*torch.ones((size,size))
    # P = (phase-phase.min())/(phase.max()-phase.min())




    img_output = diffraction.light_input_forward(img_in)
    # img_output = img_output*torch.exp(2.0j*pi*phase)



    img_output = torch.abs(img_output) ** 2
    diffraction.out_img(img_output[0, :, :])












"""空域直接计算"""
    # x1 = np.linspace(0, D, size)
    # y1 = np.linspace(0, D, size)
    # x2 = np.linspace(0, D, size)
    # y2 = np.linspace(0, D, size)
    #
    # xg1, yg1, xg2, yg2 = np.meshgrid(x1, y1, x2, y2,indexing='ij')
    # r = np.sqrt((xg1 - xg2) ** 2 + (yg1 - yg2) ** 2 + z ** 2)
    # dA = (D/size)**2   #differential area sqared
    # w = (z / r ** 2) * (1 / (2 * np.pi * r) + 1 / (1j * lam)) * np.exp(1j * 2 * np.pi * r / lam) * dA
    # weights = w.reshape(size ** 2, size ** 2)
    #
    #
    # img = np.reshape(img,(1,-1))
    # out = np.matmul(img,np.transpose(weights))
    # out_RS= np.reshape(out,(size,size))
    # out_RS = np.abs(out_RS)*np.abs(out_RS)
    # out_img()





