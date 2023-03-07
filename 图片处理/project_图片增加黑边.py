import torch
import torch.nn.functional as F
import os

import PIL.Image as Image
import torchvision.transforms as transforms
path = '/Users/hzh/Desktop/bilinear200'
target = '/Users/hzh/Desktop/bilinear_padding_245'

if os.path.exists(target) == False:
        os.makedirs(target)
img_list = os.listdir(path)
try:
    img_list.remove('.DS_Store')
except:
    pass

for img in img_list:
        img_path = os.path.join(path, img)
        target_path = os.path.join(target, img)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)
        img = torch.unsqueeze(img, dim=0)
        img_resize = F.interpolate(img, (160, 160), mode='bilinear')
        img_padding = F.pad(img_resize, (40, 40, 40, 40))
        img_padding = img_padding[0, 0, :, :].numpy() * 255
        im = img_padding.convert('L')
        im.save(target_path)
