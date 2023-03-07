#########################################################################
# path就是imagenet的tar文件路径
# save_path就是解压缩文件路径
#########################################################################
import os
import tarfile
from 装饰器.decorator_程序启动 import logit


@logit
def decompression_tar(path, save_path):
    """拨开第一层tar"""
    with tarfile.open(path) as file:
        for t1 in file.getmembers():
            file.extract(t1.name, save_path)

    img_list = os.listdir(save_path)
    for tar in img_list:
        tar_list_path = os.path.join(save_path, tar)

        with tarfile.open(tar_list_path) as file:
            for img in file.getmembers():
                img_path = os.path.join(save_path, tar[:-4])
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                file.extract(img.name, img_path)
        os.remove(tar_list_path)
    print('done')


if __name__ == '__main__':
    path = r'/Users/hzh/Downloads/ILSVRC2012_img_train_t3.tar'
    save_path = r'/Users/hzh/Downloads/dog'
    decompression_tar(path, save_path)

