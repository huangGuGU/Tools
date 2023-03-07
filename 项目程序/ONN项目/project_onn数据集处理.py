import torch
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
path = '/Users/hzh/Desktop/缩小'
target = '/Users/hzh/Desktop/缩小input'
if os.path.exists(target):
    pass
else:
    os.mkdir(target)

img_list = os.listdir(path)
try:
    img_list.remove('.DS_Store')
except:
    pass

for image in img_list:
    img_path = os.path.join(path, image)
    target_path = os.path.join(target, image)
    img = cv2.imread(img_path,0)

    label =img

    img = transforms.Compose([transforms.ToTensor(),
                              transforms.Resize((240, 240))])(img)
    label = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((120, 120))])(label)


    img = torch.unsqueeze(img, dim=0)
    label = torch.squeeze(label, dim=0).numpy()*255
    # padding_size = (label.shape[-1] - img.shape[-1]) // 2
    # img = F.pad(img, (padding_size, padding_size, padding_size, padding_size))
    # img = img.squeeze(0).squeeze(0).numpy()*255
    # label = label.squeeze(0).squeeze(0).numpy()*255
    cv2.imwrite(target_path,label)

print('done')