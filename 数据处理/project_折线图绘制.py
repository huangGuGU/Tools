import matplotlib.pyplot as plt
from 装饰器.decorator_程序启动 import logit

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


@logit
def make_chart(x_list, y_list, y2):
    plt.rcParams['savefig.dpi'] = 300
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.plot(x_list, y_list, marker='.', color='green', label='CNN')
    plt.plot(x_list, y2, marker='.', color='red', label='MLP')
    # plt.xlabel("$\sigma$")
    # plt.ylabel("SSIM")
    # plt.title(name)
    # plt.legend(loc=0, ncol=2)
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'/Users/hzh/Desktop/1.png')
    plt.show()


if __name__ == '__main__':
    # x = ["4", "10", "15", "20"]
    # y = [0.9101, 0.9194, 0.9224, 0.9235]

    x = ["0倍噪音", "0.1倍噪音", "0.2倍噪音", "0.5倍噪音"]
    y = [0.48, 3.36, 26.98, 28.95]
    y2 = [1.04, 14.02, 36.51, 119.95]

    make_chart(x, y, y2)

# 120  SSIM: 0.9277 PSNR: 20.5367 PCC:0.9603
# 220  SSIM: 0.9229 PSNR: 20.0288 PCC:0.9577
# 600  SSIM: 0.898 PSNR: 18.4219 PCC:0.9488
# 1500  SSIM: 0.8925 PSNR: 18.1631 PCC:0.9471
