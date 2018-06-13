# -*- coding:UTF-8 -*-

'''
    正则表达式
'''

import re

import log_utils as log

# re.match函数
log.loge("match函数")
# 在起始位置匹配
print(re.match("www", "www.baidu.com").span())
# 不在起始位置匹配
print(re.match("com", "www.baidu.com"))

line = "Cat are smarter than dogs"
matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)

if matchObj:
    print("matchObj.group(): ", matchObj.group())
    print("matchObj.group(1): ", matchObj.group(1))
    print("matchObj.group(2): ", matchObj.group(2))
else:
    print("No match!")
log.loge("match函数")

log.loge("search方法")
# re.search方法
# 在起始位置匹配
print(re.search("www", "www.baidu.com").span())
# 不在起始位置匹配
print(re.search("com", "www.baidu.com").span())

matchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)

if matchObj:
    print("matchObj.group(): ", matchObj.group())
    print("matchObj.group(1): ", matchObj.group(1))
    print("matchObj.group(2): ", matchObj.group(2))
else:
    print("No match!")
log.loge("search方法")

# re.match与re.search的区别
matchObj = re.match(r'dogs', line, re.M | re.I)

if matchObj:
    print("match --> matchObj.group(): ", matchObj.group())
else:
    print("No match!")

matchObj = re.search(r'dogs', line, re.M | re.I)

if matchObj:
    print("search --> matchObj.group(): ", matchObj.group())
else:
    print("No match!")

# 检索和替换
phone = "2004-959-559 # 这是一个国外电话号码"

# 删除字符串中的Python注释
log.loge("sub方法")
num = re.sub("#.*$", "", phone)
print("电话号码是: ", num)

# 删除非数字(-)的字符串
num = re.sub(r'\D', "", phone)
print("电话号码是: ", num)


# repl 参数是一个函数
# 将匹配的数字x2

def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)


s = 'A23G4HFD567'
print(re.sub('(?P<value>\d+)', double, s))
log.loge("sub方法")

log.loge("compile方法")
# re.compile 函数
# 用于匹配至少一个数字
pattern = re.compile(r'\d+')

# 查找头部,没有匹配
sample_data = 'one12twothree34four'
m = pattern.match(sample_data)
print("match 1 -> ", m)

# 从'e'的位置 开始匹配,没有匹配
m = pattern.match(sample_data, 2, 10)
print("match 2 -> ", m)

# 从'1'的位置 开始匹配，正好匹配
m = pattern.match(sample_data, 3, 10)
print("match 3 -> ", m)

print(m.group(0))
print(m.span(0))
print(m.end(0))
print(m.span(0))

"""
在上面，当匹配成功时返回一个 Match 对象，其中：
group([group1, …]) 方法用于获得一个或多个分组匹配的字符串，当要获得整个匹配的子串时，可直接使用 group() 或 group(0)；
start([group]) 方法用于获取分组匹配的子串在整个字符串中的起始位置（子串第一个字符的索引），参数默认值为 0；
end([group]) 方法用于获取分组匹配的子串在整个字符串中的结束位置（子串最后一个字符的索引+1），参数默认值为 0；
span([group]) 方法返回 (start(group), end(group))。

"""

# re.I 表示忽略大小写
pattern = re.compile(r'([a-z]+) ([a-z]+)', re.I)
m = pattern.match("Hello World Wide Web")

# 匹配成功，返回一个 Match 对象
print(m)

# 返回匹配成功的整个子串
print("m.group(0) ", m.group(0))

# 返回匹配成功的整个子串的索引

# 返回第一个分组匹配成功的子串
print("m.group(1) ", m.group(1))

# 返回第一个分组匹配成功的子串的索引
print("m.span(1) ", m.span(1))

# 返回第二个分组匹配成功的子串
print("m.group(2) ", m.group(2))

# 返回第二个分组匹配成功的子串
print("m.span(2) ", m.span(2))

# 等价于 (m.group(1), m.group(2),
print("m.groups ", m.groups())

# 返回第二个分组匹配成功的子串
# print("m.group(3) ", m.group(3))

log.loge("compile方法")

log.loge("findall")
# findall
# 在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。
# 查找数字
pattern = re.compile(r'\d+')
result1 = pattern.findall('runoob 123 google 456')
print(result1)
result2 = pattern.findall('run88oob123google456', 0, 10)
print(result2)
log.loge("findall")

# re.finditer 和 findall 类似，在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回。
log.loge("finditer")
it = re.finditer(r"\d+", "12a32bc43jf3")
for match in it:
    print(match.group())

log.loge("finditer")

# re.split split 方法按照能够匹配的子串将字符串分割后返回列表
log.loge("split")
test_data = 'runoob, runoob, runoob.'
print(re.split('\W+', test_data))
print(re.split('(\W+)', test_data))
print(re.split('\W+', ' runoob, runoob, runoob.', 1))
log.loge("split")

# 笔记
line = "Cats are smarter than dogs"
matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)
if matchObj:
    print("matchObj.group() : ", matchObj.group())
    print("matchObj.group(1) : ", matchObj.group(1))
    print("matchObj.group(2) : ", matchObj.group(2))
else:
    print("No match!!")

"""
正则表达式：
r'(.*) are (.*?) .*'
解析:
首先，这是一个字符串，前面的一个 r 表示字符串为非转义的原始字符串，让编译器忽略反斜杠，也就是忽略转义字符。但是这个字符串里没
有反斜杠，所以这个 r 可有可无。
 (.*) 第一个匹配分组，.* 代表匹配除换行符之外的所有字符。
 (.*?) 第二个匹配分组，.*? 后面多个问号，代表非贪婪模式，也就是说只匹配符合条件的最少字符
 后面的一个 .* 没有括号包围，所以不是分组，匹配效果和第一个一样，但是不计入匹配结果中。
matchObj.group() 等同于 matchObj.group(0)，表示匹配到的完整文本字符
matchObj.group(1) 得到第一组匹配结果，也就是(.*)匹配到的
matchObj.group(2) 得到第二组匹配结果，也就是(.*?)匹配到的
因为只有匹配结果中只有两组，所以如果填 3 时会报错。
"""
