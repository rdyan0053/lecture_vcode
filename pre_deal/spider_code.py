'''
    爬取东南大学讲座的验证码脚本
'''


import os
import base64
import requests
from bs4 import BeautifulSoup
import urllib3
from vcode_ddddocr_resnet.pre_deal.ids_encrypt import encryptAES
import json


def login(username, password):
    session = requests.Session()  # session会话对象用于跨请求保持请求的参数
    form = {"username": username}
    url = "https://newids.seu.edu.cn/authserver/login?goto=http://my.seu.edu.cn/index.portal"
    #  获取登录页面表单，解析隐藏值
    # 构造请求头
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.42",
        'Cookie': 'xxx换成自己的cookie',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/wxpic,image/tpg,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
    }
    urllib3.disable_warnings()  # 不然会有warning

    res = session.get(url=url, headers=headers, verify=False)  # 先进行一次登录请求以获取登录表单
    soup = BeautifulSoup(res.text, 'html.parser')  # 获得网页源码
    attrs = soup.select('[tabid="01"] input[type="hidden"]')  # 获取隐藏属性

    for k in attrs:
        if k.has_attr('name'):
            form[k['name']] = k['value']
        elif k.has_attr('id'):
            form[k['id']] = k['value']

    form['password'] = encryptAES(password, form['pwdDefaultEncryptSalt'])  # 已经从form中获取加密形式，通过encryptAES进行特定格式的加密

    # 登录认证
    session.post(url, data=form)
    # 登录ehall
    session.get('http://ehall.seu.edu.cn/login?service=http://ehall.seu.edu.cn/new/index.html')
    # 获取个人信息
    new_headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.42",
        'Cookie': 'xxx换成自己的cookie',
    }
    res = session.get('http://ehall.seu.edu.cn/jsonp/userDesktopInfo.json', headers=new_headers)
    json_res = json.loads(res.text)
    try:
        name = json_res["userName"]
        print(f"小{name[0]}同学登陆成功！\n")
    except Exception:
        print("认证失败！")
        return False
    return session


def get_verify_code(session, image_num, dataset_type):
    '''获取验证码，并存入文件，制作数据集'''
    url = "http://ehall.seu.edu.cn/gsapp/sys/yddjzxxtjappseu/modules/hdyy/vcode.do"
    vcode_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.52',
        'Cookie': 'xxx换成自己的cookie'
    }
    for i in range(image_num):      # image_num是下载验证码数量
        res = session.get(url, headers=vcode_headers, timeout=(5, 5))  # 报错，显示为bool类型
        head, encode = res.json()['datas'].strip().split(",")
        img = base64.urlsafe_b64decode(encode)
        try:
            open(f'../../dataset/{dataset_type}/{i}.jpg', 'wb').write(img)
            print(f'第{i}张图片')
        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    # 替换成你的学号username，和密码password
    session = login(username=111111111, password='password')
    # get_verify_code(session, image_num=10000, dataset_type='pre_train')     # 没到10000张图片，一共弄了2826张图片
    get_verify_code(session, image_num=500, dataset_type='pre_test')     # 一共弄了500张图片
