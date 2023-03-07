# 
# 
from 装饰器.decorator_程序启动 import logit
import os

import cv2
import PIL.Image as Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

path = r'/Users/hzh/Desktop/高度值/加工phase_modulator0.xlsx'
save = r'/Users/hzh/Desktop/高度值/加工phase_modulator0_160.xlsx'
dataFrame = pd.read_excel(path)

dataFrame = dataFrame.iloc[:,1:]
dataFrame = dataFrame.iloc[:,40:40+160]
D_center = dataFrame[40:40+160]


with pd.ExcelWriter(save) as writer:
    D_center.to_excel(writer, sheet_name='phase_modulator')
