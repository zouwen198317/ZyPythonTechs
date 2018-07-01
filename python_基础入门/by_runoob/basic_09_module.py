# -*- coding:UTF-8 -*-
import log_utils as log

'''
python 模块
'''

# 命名空间和作用域
Money = 200


def AddMoney():
    # global Money
    Money = 10
    Money += 1;


print(Money)
AddMoney()
print(Money)

# dir()函数
# dir() 函数一个排好序的字符串列表，内容是一个模块里定义过的名字。
# 返回的列表容纳了在一个模块里定义的所有模块，变量和函数

import math

content = dir(math)
print(content)
