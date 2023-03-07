from 装饰器.decorator_程序启动 import logit
import os



@logit
def copy_multiple(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list = os.listdir(path)
    for file_name in file_list:
        path_name = os.path.join(path, file_name)
        target_name = os.path.join(save_path, file_name)
        if os.path.exists(target_name):
            pass
        else:
            os.mkdir(target_name)
        img_list = os.listdir(path_name)
        for i in img_list:
            filename, stem = os.path.splitext(i)
            img_path = os.path.join(path_name, i)

            with open(img_path, 'rb') as f1:
                img = f1.read()
                for num in range(1, 11):
                    name = filename + '_' + str(num) + stem

                    target_path = os.path.join(target_name, name)
                    with open(target_path, 'wb') as f2:
                        f2.write(img)

        print(f'文件{file_name}复制完成')


if __name__ == '__main__':
    path = '/Users/hzh/Desktop/File'
    save_path = '/Users/hzh/Desktop/File_done'
    copy_multiple(path, save_path)
