# -*- coding:UTF-8 -*-
'''

'''
"""
    python Exception
"""

# 它打开一个文件，在该文件中的内容写入内容，且并未发生异常
try:
    fh = open("testfile.txt", "w")
    fh.write("这是一个测试文件,用于测试异常!")
except IOError:
    print("Error 没有找到文件或读取文件失败")
else:
    print("内容写入成功")
    fh.close()

# 它打开一个文件，在该文件中的内容写入内容，但文件没有写入权限，发生了异常
# 为了测试，把文件修改为直读的
try:
    fh = open("testfile.txt", "w")
    fh.write("这是一个测试文件，用于测试异常!!")
except IOError:
    print("Error: 没有找到文件或读取文件失败")
else:
    print("内容写入文件成功")
    fh.close()

# try-finally

# 未catch应用会挂掉
try:
    fh = open("testfile.txt", "w")
    try:
        fh.write("这是一个测试文件,用于测试异常!")
    finally:
        print("关闭文件")
        fh.close()
except IOError:
    print("Error 没有找到文件或读取文件失败")

"""
当在try块中抛出一个异常，立即执行finally块代码。
finally块中的所有语句执行后，异常被再次触发，并执行except块代码。
参数的内容不同于异常。  
"""


# 异常的参数
def temp_convert(var):
    try:
        return int(var)
    except ValueError as verr:
        print("参数没有包含数字\n", verr)


temp_convert("xyz")


# 一个异常可以是一个字符串，类或对象。 Python的内核提供的异常，大多数都是实例化的类，这是一个类的实例的参数。
# 定义一个异常非常简单，

def functionName(level):
    if level < 1:
        raise Exception("Invalid level!", level)
        # 触发异常后，后面的代码就不会再执行
        print("aaaaa")


"""
注意：为了能够捕获异常，"except"语句必须有用相同的异常来抛出类对象或者字符串。
例如我们捕获以上异常，"except"语句如下所示：
try:
    正常逻辑
except Exception,err:
    触发自定义异常    
else:
    其余代码
"""


def mye(level):
    if level < 1:
        raise Exception("Invalid level!", level)
        # 触发异常后，后面的代码就不会再执行
        print("aaaaa")


try:
    mye(0)
except Exception as err:
    print(1, err)
else:
    print(2)


# 用户自定义异常
class NetworkError(RuntimeError):
    def _init_(self, arg):
        self.args = arg


try:
    raise NetworkError("Bad hostname")
except NetworkError as e:
    print(e.args)

# 0 作为除数

try:
    1 / 0
except Exception as e:
    '''异常的父类，可以捕获所有的异常'''
    print("0不能被除")
else:
    '''保护不抛出异常的代码'''
    print("没有异常")
finally:
    print("最后总是要执行我")
