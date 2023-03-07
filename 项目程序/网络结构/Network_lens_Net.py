
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn
from torch import pi


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Diffraction(torch.nn.Module):
    def __init__(self, lam, size):
        super(Diffraction, self).__init__()
        # optical basic variable
        # light
        self.lam = lam
        self.k = 2 * pi / self.lam
        # diffraction fringe
        self.pixel_size = 450e-9

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

    def lens_forward(self, images, f):

        proportional = torch.exp(
            ((1.0j * self.k / (2 * f)) * ((self.lam * self.fu) ** 2 + (self.lam * self.fv) ** 2))
        )/(1.0j*f*self.lam)
        k_images = torch.fft.fft2(images)
        output = proportional * torch.fft.fftshift(k_images, dim=(2, 3))
        base_phase = self.k * torch.tensor(f)
        random_diffuser = -base_phase + 2 * pi * torch.rand(self.size[0], self.size[1], device=device)

        return output,  random_diffuser

    # ##################################################################################################################

# class Diffuser(torch.nn.Module):
#
#     def __init__(self, n0, n1, lam, size, scale):
#         super(Diffuser, self).__init__()
#         self.n0 = n0
#         self.n1 = n1
#         self.delta_n = self.n1 - self.n0
#         self.lam = lam
#         self.size = torch.tensor(size)
#         self.scale = scale
#
#     def diffuserplane(self, miu, sigma0, sigma):
#
#         kernel = self.gaussian_2d_kernel(
#             torch.tensor((torch.sqrt(2 * torch.log(torch.tensor(2))) * sigma) / (self.lam / self.scale),
#                          dtype=torch.int16) * 2 + 1, sigma).unsqueeze(0).unsqueeze(0)
#
#         diffuser = torch.FloatTensor(2 * pi * self.delta_n / self.lam *
#                                      torch.normal(miu, sigma0,
#                                                   size=(self.size[0] + kernel.shape[2] - 1,
#                                                         self.size[1] + kernel.shape[3] - 1))).unsqueeze(0).unsqueeze(0)
#
#         return torch.conv2d(diffuser, kernel, padding=0).squeeze(0).squeeze(0)
#
#     def gaussian_2d_kernel(self, kernel_size, Sigma):
#         kernel = torch.zeros([kernel_size, kernel_size])
#         center = kernel_size // torch.tensor(2)
#
#         x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
#         y = torch.linspace(kernel_size // 2, -kernel_size // 2, kernel_size)
#         mask_x, mask_y = torch.meshgrid(x, y, indexing='xy')
#         mask_rho = torch.hypot(mask_x, mask_y)
#         mask = (mask_rho < center)
#
#         if Sigma == 0:
#             sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
#
#         s = 2 * (Sigma ** 2)
#         for i in range(0, kernel_size):
#             for j in range(0, kernel_size):
#                 x = (i - center) * (self.lam / self.scale)
#                 y = (j - center) * (self.lam / self.scale)
#
#                 kernel[i, j] = torch.exp(torch.div(-(x ** 2 + y ** 2), s))
#
#         kernel = kernel * mask
#         kernel = kernel / torch.sum(kernel)
#
#         return torch.FloatTensor(kernel)


class lens_Net(torch.nn.Module):
    def __init__(self, num_layers, size, lam, ratio,
                 miu, sigma0, sigma, n0,
                 n1):  # 毛玻璃的参数   220目：miu=63*10**-6,sigma0=14*10**-6,sigma=15.75*10**-6
        #             600目：miu=16*10**-6,sigma0=5*10**-6,sigma=4*10**-6
        super(lens_Net, self).__init__()
        self.num_layers = num_layers

        self.size = torch.tensor(size)
        self.lam = lam

        self.n0 = n0
        self.n1 = n1
        self.miu = miu
        self.sigma0 = sigma0
        self.sigma = sigma
        self.dropout = torch.nn.Dropout(ratio)

        '''毛玻璃的参数'''
        # D = Diffuser(scale=self.scale, lam=self.lam, n0=self.n0, n1=self.n1,
        #              size=self.size).diffuserplane(miu=self.miu, sigma0=self.sigma0,
        #                                            sigma=self.sigma)
        # self.register_buffer(f'phase_input', D)

        # learning weight

        self.phase_modulator = []
        for layer in range(self.num_layers):
            self.phase_modulator.append(
                torch.nn.Parameter(torch.rand(size=(self.size[0], self.size[1]))))
            self.register_parameter(f'phase_modulator_{layer}', self.phase_modulator[layer])

    def forward(self, inputs):

        diffraction = Diffraction(lam=self.lam, size=self.size)
        f = 20e-3
        x, random_diffuser = diffraction.lens_forward(inputs, f)
        x = x * torch.exp(1.0j * random_diffuser)

        x = diffraction.light_forward(x, f)

        # ##############################################################################################################

        '''调制层'''
        for index, phase in enumerate(self.phase_modulator):
            # x = diffraction.light_forward(x,10)

            One = torch.ones((self.size[0], self.size[1]), device=device)
            dropout_mask = self.dropout(One)
            x = x * torch.exp(1.0j * 2 * pi * torch.sigmoid(phase)) * dropout_mask
            # print(f'phase_modulator_{index}:{phase}')

        # ##############################################################################################################

        '''输出层'''
        x = diffraction.light_forward(x, 20)
        x = torch.abs(x) ** 2
        x = torch.tanh(x)  # 1

        return x


if __name__ == '__main__':
    x1 = torch.rand([4, 1, 240, 240])
    label1 = torch.rand([4, 1, 240, 240])
    models = lens_Net(num_layers=1, size=(240, 240), lam=450e-9,
                      miu=0, sigma0=0, sigma=0, n0=0,
                      n1=0, ratio=0.3)

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

