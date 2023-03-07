#####################################################################################
# path路径是CIFAR-10的存放路径
# loc_1和loc_2是训练集和测试集路径
#####################################################################################
import numpy as np
import cv2
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


# 训练集有五个批次，每个批次10000个图片，测试集有10000张图片
def CIFAR10_save():
    for i in range(1, 6):
        data_name = path + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            # 变灰度图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 要改成不同的形式的文件只需要将文件后缀修改即可
            img_name = str(data_dict[b'labels'][j]) + str(i * 10000 + j) + '.jpg'
            img_path = os.path.join(loc_1, img_name)
            cv2.imwrite(img_path, img)

        print(f'{data_name} done')

    test_data_name = path + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 要改成不同的形式的文件只需要将文件后缀修改即可
        img_name = str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        img_path = os.path.join(loc_2, img_name)
        cv2.imwrite(img_path, img)
    print(f'{test_data_name} done')
    print('all done')


if __name__ == '__main__':
    loc_1 = 'train_CIFAR10'
    loc_2 = 'test_CIFAR10'
    path = '/Users/hzh/Downloads/cifar-10-batches-py'
    if not os.path.exists(loc_1):
        os.mkdir(loc_1)
    if not os.path.exists(loc_2):
        os.mkdir(loc_2)

    CIFAR10_save()
    print('done')
