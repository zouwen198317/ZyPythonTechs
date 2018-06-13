# -*- coding:UTF-8 -*-

# 数据转换

# 将x转换为一个整数
from pyparsing import unichr

f = 32.5
i = int(f)
print(i)
print(type(i))

# 将x转换到一个浮点数
fl = float(i)
print(fl)
print(type(fl))
# 创建一个复数
c = complex(3 + 1.5)
print(c)
print(type(c))

# 将对象 x 转换为字符串
s = str(f)
print(s)
print(type(s))

# 将对象x转换为表达式字符串
s_repr = repr(1 + 15 / 2)
print(s_repr)
print(type(s_repr))

# 用来计算在字符串中的有效Python表达式,并返回一个对象
s_eval = eval('1+12,3*5,4*12')
# (13, 15, 48)
print(s_eval)
# <class 'tuple'>
print(type(s_eval))

# 将序列s转换为一个元组
s_l = {12, 5, 3}
# list->tuple
s_t = tuple(s_l)
print(s_t)
print(type(s_t))

# 将序列 s 转换为一个列表
# tuple->list
s_l2 = list(s_t)
print(s_l2)
print(type(s_l2))

# 转换为可变集合
set1 = set(s_l2)
print(set1)
print(type(set1))

# 创建一个字典。d必须是一个序列(key, value)元组。
d = dict({'a': 12, 'b': 33, 'c': 'ofo'})

print(d)
print(type(d))

# 转换为不可变集合
f1 = frozenset(d)

# frozenset({'b', 'c', 'a'})
print(f1)
# <class 'frozenset'>
print(type(f1))

# 将一个整数转换为一个字符
c1 = chr(98)
print(c1)
print(type(c1))

# 将一个整数转换为Unicode字符
uc = unichr(97)
print(uc)
print(type(uc))

# 将一个字符转换为它的整数值
o1 = ord('a')
print(o1)
# 将一个整数转换为一个十六进制字符串
h1 = hex(11)
print(h1)
# 将一个整数转换为一个八进制字符串
o2 = oct(11)
print(o2)
