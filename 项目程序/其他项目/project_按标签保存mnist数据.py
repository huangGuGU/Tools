import os
import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt

path = r'/Users/hzh/Desktop/y'
target = r'/Users/hzh/Desktop/test_img'
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

for num in range(0,10):
    target_path = os.path.join(target, str(num))
    if os.path.exists(target_path):
        pass
    else:
        os.makedirs(target_path)
    for i in range(test_images.shape[0]):
        if str(test_labels[i]) == str(num):
            a = test_images[i]
            plt.imsave(target_path+f'/Testimg_{i}.png',a,cmap='gray')

print('done')

pass
