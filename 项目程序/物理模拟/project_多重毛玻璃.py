#########################################################################
# 毛玻璃分成n*n个patch，每个patch中有一个圆形的孔洞
# 仿真一个n*n的小毛玻璃拼成的大毛玻璃，相干长度L自定
# 使用的毛玻璃公式是修改过后的
#########################################################################
import matplotlib.pyplot as plt
import torch


class Diffuser(torch.nn.Module):
    def __init__(self, n0, n1, lam, size, patch_size, L):
        super(Diffuser, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.delta_n = self.n1 - self.n0
        self.lam = lam
        self.size = size
        patch_x = torch.arange(-patch_size // 2, patch_size // 2).clone().detach()
        patch_y = torch.arange(-patch_size // 2, patch_size // 2).clone().detach()
        self.patch_X, self.patch_Y = torch.meshgrid(patch_x * 1.0, patch_y * 1.0, indexing='xy')
        self.L = L

    def diffuser_plane(self, miu, sigma0, sigma):

        kernel = self.gaussian_2d_kernel(
            torch.tensor((torch.sqrt(2 * torch.log(torch.tensor(2))) * sigma) / self.lam,
                         dtype=torch.int16) * 2 + 1, sigma).unsqueeze(0).unsqueeze(0)

        W = torch.FloatTensor(2 * torch.pi * self.delta_n / self.lam * torch.normal(miu, sigma0,
                                                                                    size=(
                                                                                        self.size[0] + kernel.shape[
                                                                                            2] - 1,
                                                                                        self.size[1] + kernel.shape[
                                                                                            3] - 1))).unsqueeze(
            0).unsqueeze(0)
        h_base = 0.42 * self.lam
        W_K = torch.conv2d(W, kernel, padding=0).squeeze(0).squeeze(0) + h_base
        D_h_d = W_K % (self.lam / (1.5 - 1))
        D_phi = (1.5 - 1) * 2 * torch.pi * D_h_d / self.lam
        return D_h_d, D_phi

    def gaussian_2d_kernel(self, kernel_size, Sigma):
        kernel = torch.zeros([kernel_size, kernel_size])
        center = torch.div(kernel_size, 2, rounding_mode='floor')

        x = torch.linspace(-center, center, kernel_size)
        y = torch.linspace(center, -center, kernel_size)
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

    def mask(self, ):
        mask = torch.hypot(self.patch_X, self.patch_Y) < self.L
        return mask

    def gaussion_height(self, ):
        def function_R(L, lam, x, y):
            return torch.exp(
                -torch.pi * ((x * lam) ** 2 + (y * lam) ** 2) / (L * lam) ** 2)

        R = function_R(self.L, self.lam, self.patch_X, self.patch_Y)
        return R


if __name__ == '__main__':
    "每个patch单独生成一个毛玻璃"
    # size = 240
    # L = 3
    # patch_size = int(L * 2)
    # D = torch.zeros((size,size))
    #
    # for i in range(0, size, patch_size):
    #     for j in range(0, size, patch_size):
    #         D_h_d, D_phi = Diffuser(lam=450e-9, n0=1, n1=1.5,
    #                                 size=(patch_size, patch_size)).\
    #                                 diffuser_plane(miu=63 * 10 ** -6, sigma0=14 * 10 ** -6,sigma=15.75 * 10 ** -6)
    #         mask = Diffuser.Mask(patch_size)
    #         D_phi = D_phi*mask
    #
    #         D[i:i + patch_size, j: j + patch_size] = D_phi
    # plt.imshow(D)
    # plt.colorbar()
    # plt.show()
    def out_img(img_out, target_path):
        # 图片分辨率 = figsize*dpi 代码为512*512
        plt.rcParams['figure.figsize'] = (10.24, 10.24)
        plt.rcParams['savefig.dpi'] = 300
        # 去除白框
        plt.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # 保存图片，cmap为调整配色方案
        plt.imshow(img_out)
        plt.savefig(target_path)
        # plt.savefig(path,dpi = 300,format = 'png',pad_inches = 0,transparent = True)

        plt.show()

    "每个patch是从一个毛玻璃抠出"
    size = 240
    L = 30
    patch_size = int(L * 2)
    Phase = torch.zeros((size, size))
    Height = torch.zeros((size, size))
    D = Diffuser(lam=450e-9, n0=1, n1=1.5,
                 size=(size, size), patch_size=patch_size, L=L)
    D_h_d, D_phi = D.diffuser_plane(miu=63 * 10 ** -6, sigma0=14 * 10 ** -6,
                                    sigma=15.75 * 10 ** -6)

    for i in range(0, size, patch_size):
        for j in range(0, size, patch_size):
            mask = D.mask()
            D_phi_patch = D_phi[i:i + patch_size, j: j + patch_size] * mask
            Phase[i:i + patch_size, j: j + patch_size] = D_phi_patch
            Height[i:i + patch_size, j: j + patch_size] = D.gaussion_height()
    out_img(Phase, target_path="/Users/hzh/Desktop/ 1.png")
    plt.imshow(Phase)
    plt.colorbar()
    plt.show()

    out_img(Height, target_path="/Users/hzh/Desktop/ 2.png")
    plt.imshow(Height)
    plt.colorbar()
    plt.show()

