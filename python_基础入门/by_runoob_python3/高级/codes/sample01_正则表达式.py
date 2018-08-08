#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample01_正则表达式.py
# @Date    :   2018/8/8
# @Desc    :

import common_header
import re


def test():
    '''
    re.match函数
    re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。

    pattern	匹配的正则表达式
    string	要匹配的字符串。
    flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。参见：正则表达式修饰符 - 可选标志

    我们可以使用group(num) 或 groups() 匹配对象函数来获取匹配表达式。

    匹配对象方法	描述
    group(num=0)	匹配的整个表达式的字符串，group() 可以一次输入多个组号，在这种情况下它将返回一个包含那些组所对应值的元组。
    groups()	返回一个包含所有小组字符串的元组，从 1 到 所含的小组号。
    :return:
    '''
    # 在起始位置匹配
    print(re.match("www", "www.runoob.com").span())
    # 不在起始位置匹配
    print(re.match('com', 'www.runoob.com'))


pass


def test1():
    line = 'Cats are smrter than dogs'
    '''
    re.I 使匹配对大小写不敏感
    re.M 多行匹配，影响 ^ 和 $
    '''
    matObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)

    if (matObj):
        print("matchObj.group(): ", matObj.group())
        print("matchObj.group(1): ", matObj.group(1))
        print("matchObj.group(2): ", matObj.group(2))
    else:
        print("No match!!")
    pass


def test2():
    '''
    re.search方法
    re.search 扫描整个字符串并返回第一个成功的匹配。
    :return:
    '''
    # 在起始位置匹配
    print(re.search("www", "www.runoob.com").span())
    # 不在起始位置匹配
    print(re.search("com", "www.runoob.com").span())

    line = 'Cats are smrter than dogs'
    '''
    re.I 使匹配对大小写不敏感
    re.M 多行匹配，影响 ^ 和 $
    '''
    searchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)

    if (searchObj):
        print("searchObj.group(): ", searchObj.group())
        print("searchObj.group(1): ", searchObj.group(1))
        print("searchObj.group(2): ", searchObj.group(2))
    else:
        print("No match!!")
    pass


def test3():
    """
    re.match与re.search的区别
    re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；
    而re.search匹配整个字符串，直到找到一个匹配。
    :return:
    """
    line = 'Cats are smrter than dogs'
    '''
    re.I 使匹配对大小写不敏感
    re.M 多行匹配，影响 ^ 和 $
    '''
    matchObj = re.match(r'dogs', line, re.M | re.I)
    if matchObj:
        print("match --> matchObj.group(): ", matchObj.group())
    else:
        print("No match!!")

    searchObj = re.search(r'dogs', line, re.M | re.I)
    if searchObj:
        print("search --> searchObj.group(): ", searchObj.group())
    else:
        print("No match!!")
    pass


def test4():
    '''
    检索和替换
    Python 的re模块提供了re.sub用于替换字符串中的匹配项。

    语法：
    re.sub(pattern, repl, string, count=0)
    参数：
    pattern : 正则中的模式字符串。
    repl : 替换的字符串，也可为一个函数。
    string : 要被查找替换的原始字符串。
    count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
    :return:
    '''
    phone = '2004-959-559 # 这是一个电话号码'
    # 删除注释
    num = re.sub(r'#.*$', "", phone)
    print("电话号码: ", num)

    # 移除非数字的内容
    num = re.sub(r'\D', '', phone)
    print("电话号码: ", num)

    pass


def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)


def test5():
    """
    repl 参数是一个函数
    :return:
    """
    s = 'A23G4HFD567'
    # 将字符串中的匹配的数字乘于 2
    print(re.sub('(?P<value>\d+)', double, s))
    pass


def test6():
    """
    compile 函数
    compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用。

    语法格式为：
    re.compile(pattern[, flags])
    参数：
    pattern : 一个字符串形式的正则表达式
    flags 可选，表示匹配模式，比如忽略大小写，多行模式等，具体参数为：
    re.I 忽略大小写
    re.L 表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境
    re.M 多行模式
    re.S 即为' . '并且包括换行符在内的任意字符（' . '不包括换行符）
    re.U 表示特殊字符集 \w, \W, \b, \B, \d, \D, \s, \S 依赖于 Unicode 字符属性数据库
    re.X 为了增加可读性，忽略空格和' # '后面的注释

    :return:
    """
    # 用于匹配至少一个数字
    pattern = re.compile(r'\d+')
    # 查找头部，没有匹配
    sample_str = 'one12twothree34four'
    m = pattern.match(sample_str)
    print(m)

    # 从'e'的位置开始匹配，没有匹配
    m = pattern.match(sample_str, 2, 10)
    print(m)

    # 从1的位置开始匹配，正好匹配
    m = pattern.match(sample_str, 3, 10)
    '''
    匹配对象:
    group([group1, …]) 方法用于获得一个或多个分组匹配的字符串，当要获得整个匹配的子串时，可直接使用 group() 或 group(0)；
    start([group]) 方法用于获取分组匹配的子串在整个字符串中的起始位置（子串第一个字符的索引），参数默认值为 0；
    end([group]) 方法用于获取分组匹配的子串在整个字符串中的结束位置（子串最后一个字符的索引+1），参数默认值为 0；
    span([group]) 方法返回 (start(group), end(group))。
    '''
    # 匹配成功，返回一个 Match 对象
    print(m)
    # 匹配成功返回整个子串
    print(m.group(0))
    print(m.start(0))
    print(m.end(0))
    print(m.span(0))

    pattern = re.compile(r'([a-z]+) ([a-z]+)', re.I)
    m = pattern.match('Hello World Wide Web')
    # 匹配成功，返回一个 Match 对象
    print(m)
    # 匹配成功返回整个子串
    print(m.group(0))
    # 返回第一个分组匹配成功的子串
    print(m.group(1))
    # 返回第二个分组匹配成功的子串
    print(m.group(2))
    # 返回匹配成功的整个子串的索引
    print(m.span(0))
    # 返回第一个分组匹配成功的子串的索引
    print(m.span(1))
    # 返回第二个分组匹配成功的子串的索引
    print(m.span(2))

    # 等价于(m.group(1),m.group(2),)
    print(m.groups())

    print(m.group(3))
    pass


def test7():
    '''
    findall
    在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。

    注意： match 和 search 是匹配一次 findall 匹配所有。

    语法格式为：
    findall(string[, pos[, endpos]])
    参数：
    string 待匹配的字符串。
    pos 可选参数，指定字符串的起始位置，默认为 0。
    endpos 可选参数，指定字符串的结束位置，默认为字符串的长度。
    :return:
    '''
    pattern = re.compile(r'\d+')
    result1 = pattern.findall('runoob 123 google 456')
    result2 = pattern.findall('run88oob123google456', 0, 10)
    print(result1)
    print(result2)
    pass


def test8():
    '''
    re.finditer
    和 findall 类似，在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回。
    re.finditer(pattern, string, flags=0)
    参数：
    参数	描述
    pattern	匹配的正则表达式
    string	要匹配的字符串。
    flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。参见：正则表达式修饰符 - 可选标志
    :return:
    '''
    it = re.finditer(r'\d+', '12a32bc43jf3')
    for match in it:
        print(match.group())
    pass


def test9():
    """
    re.split
    split 方法按照能够匹配的子串将字符串分割后返回列表，它的使用形式如下：
        re.split(pattern, string[, maxsplit=0, flags=0])
    参数：
        参数	描述
        pattern	匹配的正则表达式
        string	要匹配的字符串。
        maxsplit	分隔次数，maxsplit=1 分隔一次，默认为 0，不限制次数。
        flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。参见：正则表达式修饰符 - 可选标志

    :return:
    """
    print(re.split('\W+', 'runoob,runoob,runoob.'))
    print(re.split('(\W+)', ' runoob,runoob,runoob.'))
    print(re.split('\W+', ' runoob,runoob,runoob.', 1))
    pass


if __name__ == '__main__':
    # test()
    # test1()
    # test2()
    # test3()
    '''
    re.I 忽略大小写
    re.L 表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境
    re.M 多行模式
    re.S 即为' . '并且包括换行符在内的任意字符（' . '不包括换行符）
    re.U 表示特殊字符集 \w, \W, \b, \B, \d, \D, \s, \S 依赖于 Unicode 字符属性数据库
    re.X 为了增加可读性，忽略空格和' # '后面的注释
    '''
    # test4()
    # test5()
    # test6(
    # test7()
    # test8()
    test9()
