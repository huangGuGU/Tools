import torch
import torch.nn.functional as F
import os
import PIL.Image as Image
import torchvision.transforms as transforms


def img_padding(path, save):
    if not os.path.exists(save):
        os.makedirs(save)
    img_list = os.listdir(path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass

    for img in img_list:
        img_path = os.path.join(path, img)
        save_path = os.path.join(save, img)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)
        img = torch.unsqueeze(img, dim=0)
        img_resize = F.interpolate(img, (160, 160), mode='bilinear')
        padding = F.pad(img_resize, (40, 40, 40, 40))
        padding = padding[0, 0, :, :].numpy() * 255
        im = padding.convert('L')
        im.save(save_path)


if __name__ == '__main__':
    file_path = ''
    Save_path = ''
    img_padding(file_path, Save_path)
