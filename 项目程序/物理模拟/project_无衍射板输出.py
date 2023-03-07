import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
import torch.nn
from torch import pi
from torchvision.transforms import transforms


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
        # output = torch.fft.ifftshift(output)
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
        phase = 2 * torch.pi * phase
        self.register_buffer(f'diffuser', phase)

        # learning weight
        self.phase_modulator = []
        for layer in range(self.num_layers):
            self.phase_modulator.append(
                torch.nn.Parameter(torch.rand(size=(self.size[0], self.size[1]))))
            self.register_parameter(f'phase_modulator_{layer}', self.phase_modulator[layer])

        self.phase_output = torch.nn.Parameter(2 * pi * torch.rand(size=(self.size[0], self.size[1])))

    def forward(self, inputs, i):
        diffraction = Diffraction(lam=self.lam, size=self.size)
        x = diffraction.light_forward(inputs, 53)

        # x = x * torch.exp(1.0j * self.diffuser)

        # ################################################################################################################################################################################################################################

        '''调制层'''

        for index, phase in enumerate(self.phase_modulator):
            x = diffraction.light_forward(x, 2.7)

        # ################################################################################################################################################################################################################################

        '''输出层'''

        def out_img(img_out, target_path):
            # 图片分辨率 = figsize*dpi 代码为512*512
            plt.rcParams['figure.figsize'] = (10.24, 10.24)
            plt.rcParams['savefig.dpi'] = 300
            # 去除白框
            plt.axis('off')
            plt.margins(0, 0)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # 保存图片，cmap为调整配色方案
            plt.imshow(img_out, cmap=plt.cm.gray)
            plt.savefig(target_path)
            # plt.savefig(path,dpi = 300,format = 'png',pad_inches = 0,transparent = True)

            plt.show()

        x = diffraction.light_forward(x, 903)
        x = torch.abs(x) ** 2
        x = torch.tanh(x)  # 1
        out_img((x[0, 0, :, :].cpu().numpy()), fr'/Users/hzh/Desktop/{i}.png')

        return x


if __name__ == '__main__':
    for i in ["1"]:
        x1 = Image.open(f'/Users/hzh/Desktop/Test_{i}.png')
        x1 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((240, 240))])(x1)
        x1 = x1.unsqueeze(0)
        label1 = x1

        models = incoherent_light_Net(num_layers=4, size=(240, 240), lam=632e-9,
                                      miu=16 * 10 ** -6, sigma0=5 * 10 ** -6, sigma=4 * 10 ** -6

, n0=1,
                                      n1=1.5, ratio=0.3)
        prediction1 = models(x1, i)


