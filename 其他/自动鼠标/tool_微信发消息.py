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
    def mouseClick(lOrR, img_list):
        while True:
            if len(img_list)>1:

                location1 = pg.locateCenterOnScreen(img_list[0], confidence=0.9)
                location2 = pg.locateCenterOnScreen(img_list[1], confidence=0.9)

                if location1 is None and location2 is None:
                    img_name = img_list[0][2:-4]
                    logging.error(f'未找到图片:{img_name},请3s后再试')
                    time.sleep(3)
                else:
                    try:
                        pg.click(location1.x * 1680 / 3360, location1.y * 1050 / 2100, button=lOrR)
                        break
                    except:
                        pg.click(location2.x * 1680 / 3360, location2.y * 1050 / 2100, button=lOrR)
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


class Send(Click):
    def __init__(self, ):
        self.img_wechat_app = './wechat.png'
        self.img_object = './object.png'
        self.img_object_e = './object_e.png'
        self.img_object_home1 = './object_home1.png'
        self.img_object_home2 = './object_home2.png'
        self.img_frame1 = './frame1.png'
        self.img_frame2 = './frame2.png'
        self.img_message = './message.png'

    @logit
    def wechat(self, ):
        click.search_app('weix')
        click.mouseClick(lOrR='left', img_list=[self.img_object_home1,self.img_object_e])  # 找到聊天对象
        click.mouseClick(lOrR='left', img_list=[self.img_frame1,self.img_frame2])  # 找到目标聊天框
        pg.hotkey('command', 'v')
        pg.hotkey('enter')


class Find(Click):
    def __init__(self, ):
        self.img_safari_app = './safari.png'
        self.img_create_page1 = './create_page1.png'
        self.img_create_page2 = './create_page2.png'
        self.url_wx = 'http://www.weather.com.cn/weather1d/101190201.shtml#input'
        self.url_hf = 'http://www.weather.com.cn/weather1d/101220101.shtml#input'

    @logit
    def weather(self, location):

        click.search_app('safari')
        click.mouseClick(lOrR='left', img_list=[self.img_create_page1,self.img_create_page2])  # 新建页面

        # 进入网站
        if location == '无锡':
            URL = self.url_wx
        elif location == '合肥':
            URL = self.url_hf
        else:
            raise ValueError('location请输入无锡或者合肥')

        pg.typewrite(URL)
        pg.hotkey('enter')
        pg.hotkey('enter')
        time.sleep(4)
        click.img_cut()  # 截图 并保存


if __name__ == '__main__':

    while True:
        # SetTime()
        pg.moveTo(1374, 899, duration=0, tween=pg.easeInOutQuad)

        click = Click()

        find = Find()
        find.weather(location='合肥')

        send = Send()
        send.wechat()
        print('done')
        break
