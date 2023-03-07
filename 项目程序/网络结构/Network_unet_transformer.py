import math
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, double=True):
        super().__init__()
        if double:
            self.conv1 = DoubleConv(in_channels, in_channels * 2)
            self.down = nn.MaxPool2d(2)
            self.conv2 = DoubleConv(in_channels * 2, out_channels)
        else:
            self.conv1 = SingleConv(in_channels, in_channels * 2)
            self.down = nn.MaxPool2d(2)
            self.conv2 = SingleConv(in_channels * 2, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        x = self.conv2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.conv1 = DoubleConv(in_channels, out_channels // 2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv2 = DoubleConv(out_channels // 2, out_channels)
        else:
            self.conv1 = DoubleConv(in_channels, out_channels // 2)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv2 = DoubleConv(out_channels // 2, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        return x


class Norm(nn.Module):
    def __init__(self, batch, channel, size, eps=1e-5, affine=True, mode=None):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.mode = mode

        if self.mode == 'batch':
            self.bn = nn.BatchNorm2d(channel, eps=self.eps, affine=self.affine)
        elif self.mode == 'layer':
            self.ln = nn.LayerNorm(channel, eps=self.eps, elementwise_affine=self.affine)
        elif self.mode == 'channel':
            self.g = nn.Parameter(torch.ones(batch))
            self.b = nn.Parameter(torch.zeros(batch))
        elif self.mode == 'allin':
            self.g = nn.Parameter(torch.ones([channel, size, size]))
            self.b = nn.Parameter(torch.zeros([channel, size, size]))
        else:
            self.g = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.mode == 'batch':
            x = self.bn(x)
        elif self.mode == 'layer':
            x = self.ln(x)
        else:
            mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
            std = torch.std(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            if self.affine:
                if self.mode == 'channel':
                    shape = [1, -1] + [1] * (x.dim() - 2)
                    x = (x - mean) / (std ** 2 + self.eps).sqrt() * self.g.reshape(*shape) + self.b.reshape(*shape)
                elif self.mode == 'allin':
                    x = (x - mean) / (std ** 2 + self.eps).sqrt() * self.g + self.b
                else:
                    x = (x - mean) / (std ** 2 + self.eps).sqrt() * self.g + self.b
            else:
                x = (x - mean) / (std ** 2 + self.eps).sqrt()
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, channel):
        # 给特征图中的每个元素位置编码
        super(PositionalEncoding2D, self).__init__()
        channel = int(np.ceil(channel / 2))  # 向正无穷取整
        self.channels = channel
        inv_freq = 1. / (10000 ** (torch.arange(0, channel, 2).float() / channel))
        self.register_buffer('inv_freq', inv_freq)  # 给模型添加不可训练的参数buffer到内存register中

    def forward(self, tensor):
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4 dims!")

        bs, ch, x, y = tensor.shape
        bs, ch, x, y = torch.tensor([bs, ch, x, y])
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, 0:self.channels] = emb_x  # [h, w, c]
        emb[:, :, self.channels:2 * self.channels] = emb_y  # [h, w, c]
        x = emb[None, :, :, 0:ch].repeat(bs, 1, 1, 1).permute(0, 3, 1, 2)  # [b, c, h, w]
        return x


class MultiHeadDense(nn.Module):
    def __init__(self, channel, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(channel, channel))
        if bias:
            # raise NotImplementedError()  ## 使子类不重写直接调用时会抛出异常
            self.bias = nn.Parameter(torch.Tensor(channel, channel))
        else:
            self.register_parameter('bias', None)  # 给模型添加可训练参数parameter到内存register中
        self.initialize_parameters()

    def initialize_parameters(self):
        # 权重初始化可训练的参数parameter
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.xavier_uniform_(self.weight, gain=1.)
        if self.bias is not None:
            # 计算当前网络层输入的神经元个数
            dimensions = self.weight.dim()
            if dimensions < 2:
                raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
            num_input_maps = self.weight.size()[1]
            receptive_field_size = 1
            if self.weight.dim() > 2:
                receptive_field_size = self.weight[0][0].numel()
            fan_in = num_input_maps * receptive_field_size

            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # .bmm()输入必须有三个维度是批量矩阵乘法,其中.repeat()将权重复制成3维,权重融合特征图中每个元素的全部通道
        x = torch.bmm(x, self.weight.repeat(x.size()[0], 1, 1))  # 最后一个维度就是multihead个权重叠在一起
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channel, head, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = torch.tensor(head)

        self.pe = PositionalEncoding2D(channel)
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(channel, channel)

    def forward(self, x):
        b, c, h, w = x.size()
        b, c, h, w = torch.tensor([b, c, h, w])
        pe = self.pe(x)

        x = x + pe
        # pixel is token and channel is ndim to find relative network in between pixel
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, c]

        q = self.query(x).reshape(b, h * w, self.head, c // self.head).permute(0, 2, 1, 3)  # [b, head, h*w, d]
        k = self.key(x).reshape(b, h * w, self.head, c // self.head).permute(0, 2, 3, 1)  # [b, head, d, h*w]
        v = self.value(x).reshape(b, h * w, self.head, c // self.head).permute(0, 2, 1, 3)  # [b, head, h*w, d]
        a = self.dropout(self.softmax(torch.matmul(q, k) / math.sqrt(c // self.head)))  # [b, head, h*w, h*w]

        x = self.projection(torch.matmul(a, v).permute(0, 2, 1, 3).reshape(b, h * w, c))  # [b, h*w, c]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)  # [b, c, h, w]
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_channel, e_channel, head, dropout):
        super(MultiHeadCrossAttention, self).__init__()
        self.head = torch.tensor(head)

        self.d_pe = PositionalEncoding2D(d_channel)
        self.e_pe = PositionalEncoding2D(e_channel)
        self.query = MultiHeadDense(d_channel, bias=False)
        self.key = MultiHeadDense(d_channel, bias=False)
        self.value = MultiHeadDense(d_channel, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_channel, d_channel)

    def forward(self, d, e):
        db, dc, dh, dw = d.size()
        db, dc, dh, dw = torch.tensor([db, dc, dh, dw])
        eb, ec, eh, ew = e.size()
        eb, ec, eh, ew = torch.tensor([eb, ec, eh, ew])

        d_pe = self.d_pe(d)
        d = d + d_pe
        e_pe = self.e_pe(e)
        e = e + e_pe

        q = self.query(d.reshape(db, dc, dh * dw).permute(0, 2, 1))  # [db, dh*dw, dc]
        q = q.reshape(db, dh * dw, self.head, dc // self.head).permute(0, 2, 1, 3)  # [db, head, dh*dw, d]
        k = self.key(e.reshape(eb, dc, dh * dw).permute(0, 2, 1))  # [eb, dh*dw, dc]
        k = k.reshape(eb, dh * dw, self.head, dc // self.head).permute(0, 2, 3, 1)  # [eb, head, d, dh*dw]
        v = self.value(e.reshape(eb, dc, dh * dw).permute(0, 2, 1))  # [eb, dh*dw, dc]
        v = v.reshape(eb, dh * dw, self.head, dc // self.head).permute(0, 2, 1, 3)  # [eb, head, dh*dw, d]
        a = self.dropout(self.softmax(torch.matmul(q, k) / math.sqrt(dc // self.head)))  # [(db=eb), head, dh*dw, dh*dw]

        x = self.projection(torch.matmul(a, v).permute(0, 2, 1, 3).reshape(db, dh * dw, dc))  # [db, dh*dw, dc]
        x = x.permute(0, 2, 1).reshape(db, dc, dh, dw)  # [db, dc, dh, dw]
        return x


class EncoderBlock(nn.Module):
    def __init__(self, batch, channel, size, head, dropout):
        super(EncoderBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(channel, head, dropout)
        self.pw_ffn = nn.Sequential(
            nn.Linear(channel, 4 * channel),
            nn.ReLU(),
            nn.Linear(4 * channel, channel),
        )
        self.ln1 = Norm(batch, channel, size, mode='batch')
        self.ln2 = Norm(batch, channel, size, mode='batch')
        self.ln3 = Norm(batch, channel, size, mode='batch')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.mhsa(self.ln1(x))
        x = x + self.dropout(a)

        # position-wise feedforward network
        p = self.pw_ffn(self.ln2(x).reshape(x.size()[0], x.size()[1], -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(
            x.size())
        x = self.ln3(x + self.dropout(p))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, batch, d_channel, e_channel, size, head, dropout):
        super(DecoderBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(d_channel, head, dropout)
        self.mhca = MultiHeadCrossAttention(d_channel, e_channel, head, dropout)
        self.pw_ffn = nn.Sequential(
            nn.Linear(d_channel, 4 * d_channel),
            nn.ReLU(),
            nn.Linear(4 * d_channel, d_channel),
        )
        self.ln1 = Norm(batch, d_channel, size, mode='batch')
        self.ln2 = Norm(batch, d_channel, size, mode='batch')
        self.ln3 = Norm(batch, e_channel, size, mode='batch')
        self.ln4 = Norm(batch, d_channel, size, mode='batch')
        self.ln5 = Norm(batch, d_channel, size, mode='batch')
        self.dropout = nn.Dropout(dropout)

    def forward(self, d, e):
        d_sa = self.mhsa(self.ln1(d))
        d = d + self.dropout(d_sa)

        # 对应的编码层输出与上一层输出做交叉注意力
        d_ca = self.mhca(self.ln2(d), self.ln3(e))
        d = d + self.dropout(d_ca)

        p = self.pw_ffn(self.ln4(d).reshape(d.size()[0], d.size()[1], -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(d.size())
        x = self.ln5(d + self.dropout(p))
        return x


class DecoderBlock1(nn.Module):
    def __init__(self, batch, d_channel, e_channel, size, head, dropout):
        super(DecoderBlock1, self).__init__()
        self.mhsa = MultiHeadSelfAttention(d_channel, head, dropout)

        self.pw_ffn = nn.Sequential(
            nn.Linear(d_channel, 4 * d_channel),
            nn.ReLU(),
            nn.Linear(4 * d_channel, d_channel),
        )
        self.ln1 = Norm(batch, d_channel, size, mode='batch')
        self.ln2 = Norm(batch, d_channel, size, mode='batch')
        self.ln3 = Norm(batch, e_channel, size, mode='batch')
        self.ln4 = Norm(batch, d_channel, size, mode='batch')
        self.ln5 = Norm(batch, d_channel, size, mode='batch')
        self.dropout = nn.Dropout(dropout)

    def forward(self, d, e):
        d_sa = self.mhsa(self.ln1(d))
        d = d + self.dropout(d_sa)
        p = self.pw_ffn(self.ln4(d).reshape(d.size()[0], d.size()[1], -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(d.size())
        x = self.ln5(d + self.dropout(p))
        return x



class TransformerUp(nn.Module):
    def __init__(self, batch, label_channel, d_channel, e_channel, size, head, dropout):
        super(TransformerUp, self).__init__()
        self.conv1 = DoubleConv(label_channel, d_channel)
        self.db = DecoderBlock(batch, d_channel, e_channel, size, head, dropout)
        self.up_d = Up(d_channel, int(e_channel / 2), bilinear=True)
        self.up_e = Up(d_channel, int(e_channel / 2), bilinear=True)
        self.conv2 = DoubleConv(e_channel, int(e_channel / 2))

    def forward(self, d, e):
        # for idx, img_e in enumerate(d[0, :, :, :].detach().cpu()):
        #     plt.imsave(f'桌面/1202/{idx}.png', img_e, dpi=300)

        if e.shape[1] == 1:
            # 考虑使用自回归
            e = transforms.Resize((d.shape[2], d.shape[3]))(e)
            mask = torch.tensor(np.triu(np.ones((d.shape[2], d.shape[3])), k=0), device=e.device, dtype=e.dtype)
            e = mask * e
            # a = e.detach().cpu().numpy()[0, 0, :, :]
            # plt.imshow(a)
            # plt.show()
            e = self.conv1(e)
            # for idx, img_e in enumerate(e[0, :, :, :].detach().cpu()):
            #     plt.imsave(f'桌面/1202/{idx}.png', img_e, dpi=300)

        x = self.db(d, e)
        x = torch.cat([self.up_d(d), self.up_e(x)], dim=1)
        x = self.conv2(x)
        return x


class TransformerUp1(nn.Module):
    def __init__(self, batch, label_channel, d_channel, e_channel, size, head, dropout):
        super(TransformerUp1, self).__init__()
        self.conv1 = DoubleConv(label_channel, d_channel)
        self.db1 = DecoderBlock1(batch, d_channel, e_channel, size, head, dropout)
        self.up_d = Up(d_channel, int(e_channel / 2), bilinear=True)
        self.up_e = Up(d_channel, int(e_channel / 2), bilinear=True)
        self.conv2 = DoubleConv(e_channel, int(e_channel / 2))

    def forward(self, d, e):
        # for idx, img_e in enumerate(d[0, :, :, :].detach().cpu()):
        #     plt.imsave(f'桌面/1202/{idx}.png', img_e, dpi=300)


        x = self.db1(d, e)
        x = torch.cat([self.up_d(d), self.up_e(x)], dim=1)
        x = self.conv2(x)
        return x

class UnetTransformer(nn.Module):
    def __init__(self, batch, in_channel, out_channel, data_size, decline):
        super(UnetTransformer, self).__init__()
        self.batch = batch
        self.size = data_size[-1]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.decline = decline

        self.inconv = DoubleConv(self.in_channel, int(64 / self.decline))

        self.down1 = Down(int(64 / self.decline), int(128 / self.decline), double=True)
        self.eb1 = EncoderBlock(
            self.batch,
            int(128 / self.decline),
            int(self.size / 2),
            8,
            0.25
        )

        self.down2 = Down(int(128 / self.decline), int(256 / self.decline), double=True)
        self.eb2 = EncoderBlock(
            self.batch,
            int(256 / self.decline),
            int(self.size / 4),
            8,
            0.25
        )

        self.down3 = Down(int(256 / self.decline), int(512 / self.decline), double=True)
        self.eb3 = EncoderBlock(
            self.batch,
            int(512 / self.decline),
            int(self.size / 8),
            8,
            0.25
        )

        self.up1 = TransformerUp1(
            self.batch,
            self.out_channel,
            int(512 / self.decline),
            int(512 / self.decline),
            int(self.size / 8),
            8,
            0.25
        )

        self.up2 = TransformerUp(
            self.batch,
            self.out_channel,
            int(256 / self.decline),
            int(256 / self.decline),
            int(self.size / 4),
            8,
            0.25
        )

        self.up3 = TransformerUp(
            self.batch,
            self.out_channel,
            int(128 / self.decline),
            int(128 / self.decline),
            int(self.size / 2),
            8,
            0.25
        )

        self.outconv = SingleConv(int(64 / self.decline), self.out_channel)
        self.linear = nn.Linear(128*128,10)
        self.flatten = nn.Flatten()

    def forward(self, x, y):
        x1 = self.inconv(x)  # 32
        x2 = self.eb1(self.down1(x1))  # 64
        x3 = self.eb2(self.down2(x2))  # 128
        x4 = self.eb3(self.down3(x3))  # 256

        x = self.up1(x4, y)  # 128
        x = self.up2(x, x3)  # 64
        x = self.up3(x, x2)  # 32
        x = self.outconv(x)  # 1
        x = self.flatten(x)
        x = self.linear(x)

        outputs = F.softmax(x, dim=1)
        # outputs = torch.sigmoid(x)
        return outputs


if __name__ == '__main__':
    model = UnetTransformer(1, 1, 1, [128, 128], 2)
    data = torch.rand(1, 1, 128, 128)
    label = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    output = model(data, label)

    pass