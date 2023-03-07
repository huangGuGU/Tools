# 
# 
from 装饰器.decorator_程序启动 import logit
import os



import pyautogui as pg
import time
from 装饰器.decorator_程序启动 import *


# def SetTime():
#     Time = input('设置目标时间（xx:xx）：')
#     time_now = time.strftime("%H:%M", time.localtime())
#     if time_now-Time
#
#     if time_now == Time:  # 设置要执行的时间

class Click:

    @staticmethod
    def mouseClick(lOrR, img):
        while True:
            if True:

                location1 = pg.locateCenterOnScreen(img, confidence=0.9)


                if location1 is None :

                    logging.error(f'未找到图片,请3s后再试')
                    time.sleep(3)
                else:
                    try:
                        pg.click(location1.x * 1680 / 3360, location1.y * 1050 / 2100, button=lOrR)
                        break
                    except:

                        break

    @staticmethod
    def img_cut():
        pg.hotkey('shift', 'command', 'z')
        pg.mouseDown(212, 222)
        pg.mouseUp(890, 874)
        pg.hotkey('enter')
        time.sleep(0.1)

    @staticmethod
    def search_app(name):
        pg.hotkey('command', 'space', interval=0.1)
        pg.typewrite(name)
        pg.hotkey('enter')
        time.sleep(0.1)
        pg.hotkey('enter')

    @staticmethod
    def exit():
        pg.hotkey( 'command', 'w')

        time.sleep(0.1)








if __name__ == '__main__':

    for i in range(567):
        # time.sleep(2)
        img_next = '/Users/hzh/Desktop/next.png'
        img_download = '/Users/hzh/Desktop/download.png'
        img_page = '/Users/hzh/Desktop/page.png'
        click = Click()
        click.mouseClick('left',img_download)
        time.sleep(5)
        click.exit()
        # time.sleep(0.2)
        pg.click(300, 300)
        click.exit()

        # time.sleep(0.2)
        click.mouseClick('left',img_next)



print('done')


