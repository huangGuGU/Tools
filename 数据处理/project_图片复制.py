import os


# path = '/Users/hzh/Desktop/output'
# target = '/Users/hzh/Desktop/label'
# save = '/Users/hzh/Desktop/result/label'
#
# img_list = os.listdir(path)
# label_list = os.listdir(target)
# for i in img_list:
#     if i in label_list:
#         target_path = os.path.join(target,i)
#         save_path = os.path.join(save, i)
#         with open(target_path,'rb') as f1:
#             image = f1.read()
#             with open(save_path,'wb') as  f2:
#                 f2.write(image)
# print('done')




# """作用：第二个文件夹数字接上data文件夹的数字排序"""
# path = '/Users/hzh/Desktop/train_img_input_240'
# data = '/Users/hzh/Desktop/data'
# save = '/Users/hzh/Desktop/data'
#
# num_list = os.listdir(path)
# try:
#     num_list.remove('.DS_Store')
# except:
#     pass
# for num in num_list:
#     img_path = os.path.join(path,num)
#     save_path = os.path.join(save,num)
#     data_path = os.path.join(data, num)
#     try:
#         Data = os.listdir(data_path)
#         try:
#             Data.remove('.DS_Store')
#         except:
#             pass
#         data_number = len(Data)
#     except:
#         data_number = 0
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     imgs = os.listdir(img_path)
#     try:
#         imgs.remove('.DS_Store')
#     except:
#         pass
#
#     img_num = len(imgs)
#     for index,img in enumerate ( imgs):
#         filename, extension = os.path.splitext(img)
#         Img = os.path.join(img_path,img)
#
#         Save_path = os.path.join(save_path,f'Test_{data_number+index+1}.png')
#
#         with open(Img, 'rb') as f1:
#                 image = f1.read()
#                 with open(Save_path,'wb') as f2:
#                     f2.write(image)
#
#
# print('done')




import cv2

path = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/内窥镜/肠道书和图片/上下消化道白光/c4.2'
save = r'/Users/hzh/Desktop/训练/c4'
img_list = os.listdir(path)
try:
    img_list.remove('.DS_Store')
except:
    pass
if not os.path.exists(save):
    os.makedirs(save)
for index,img in enumerate(img_list):
    img_path = os.path.join(path,img)
    save_path =os.path.join(save,str(index)+'z'+'.jpg')
    img = cv2.imread(img_path)

    cv2.imwrite(save_path,img)

