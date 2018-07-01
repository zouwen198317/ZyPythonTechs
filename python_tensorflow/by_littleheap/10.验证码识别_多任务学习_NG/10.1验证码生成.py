# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   10.1验证码生成.py
# @Date    :   2018/7/1
# @Desc    : 需要安装captcha

import common_header

# 验证码生成库
from captcha.image import ImageCaptcha

import numpy as np

from PIL import Image

import random
import sys

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 数量少于10000，因为是随机数生成原则会有重名
num = 10000


# char_set参数可用+号连接新内容，captcha_size参数可以更改成自己想要的结果
def random_captcha_text(char_set=number, captcha_size=4):
    # 验证码列表
    captcha_text = []
    for i in range(captcha_size):
        # 随机选择
        c = random.choice(char_set)
        # 加入验证码列表
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def get_captcha_txt_and_image():
    image = ImageCaptcha()
    # 获取随机生成的验证码
    captcha_text = random_captcha_text()
    # 把验证码转换为字符串
    captcha_text = ''.join(captcha_text)
    # 生成验证码
    captcha = image.generate(captcha_text)
    import os
    if not os.path.exists("images"):
        os.makedirs("images")
    image.write(captcha_text, 'images/' + captcha_text + '.jpg')  # 写到文件


if __name__ == '__main__':
    for i in range(num):
        get_captcha_txt_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('生成完毕')
