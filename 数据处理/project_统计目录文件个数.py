#########################################################################
#
# 输入文件名,查找文件名下文件夹中的文件个数
#
#########################################################################
import os
from 装饰器.decorator_程序启动 import logit


@logit
def countFileNum(path):
    num_list = os.listdir(path)
    try:
        num_list.remove('.DS_Store')
    except:
        pass
    num_list.sort()
    total = 0
    for num in num_list:
        num_path = os.path.join(path, num)
        img_list = os.listdir(num_path)
        try:
            img_list.remove('.DS_Store')
        except:
            pass
        total += len(img_list)
        print(f'{num}: {len(img_list)}')
    print('总数是：', total)


if __name__ == '__main__':
    file_path = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/内窥镜/肠道书和图片/下消化道/下消化道白光图片/原图'
    countFileNum(file_path)
