import requests
from lxml import etree
path = 'img'





url='https://live.photoplus.cn/live/pc/98541633/#/live'
header ={'User-Agent':
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.41 Safari/537.36 Edg/101.0.1210.32'}

resp = requests.get(url,headers=header)

a = resp.request.headers

tree = etree.HTML(resp.text)


download_link = tree.xpath("/html/body/div/div/div[1]/div[1]/div[1]/div[1]/div/ul/li[1]")

img_resp = requests.get(download_link)

img = img_resp.content  # 获取的是字节

with open(path, 'wb') as f:
    f.write(img_resp.content)



resp.close()

pass