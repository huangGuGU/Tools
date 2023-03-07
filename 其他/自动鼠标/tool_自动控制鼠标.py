import pyautogui
import pyperclip
import time

img_start1 = './start1.png'
img_start2 = './start2.png'
img_pause = './pause.png'
img_app = './app.png'
location_app = pyautogui.locateCenterOnScreen(img_app, confidence=0.9)
def mouseClick( lOrR, img):
    while True:
        location = pyautogui.locateCenterOnScreen(img, confidence=0.9)
        if location is not None:
            pyautogui.click(location.x, location.y,  interval=0.3, duration=0, button=lOrR)
            break
        else:
            pyautogui.click(location_app.x, location_app.y, interval=0.3, duration=0, button=lOrR)
            time.sleep(0.5)


mouseClick( lOrR='left', img=img_start1)
for i in range(50):
    time.sleep(30)
    mouseClick(lOrR='left', img=img_pause)
    time.sleep(3)
    mouseClick( lOrR='left', img=img_start1)