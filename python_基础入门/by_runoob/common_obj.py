# -----------------     基础类 定义     -----------------
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __del__(self):
        class_name = self.__class__.__name__
        print(class_name, "销毁")


# -----------------     基础类 定义     -----------------

# -----------------     子父类 定义     -----------------
class Parent:
    '基类'
    parentAttr = 100

    def __init__(self):
        print("调用父类构造函数")

    def parentMethod(self):
        print('调用父类方法')

    def setAttr(self, attr):
        print('调用setAttr')
        Parent.parentAttr = attr

    def getAttr(self):
        print('getAttr')
        print('调用父类属性: ', Parent.parentAttr)


# 定义子类
class Child(Parent):
    def __init__(self):
        print("调用子类构造方法")

    def childMethod(self):
        print("调用子类方法")


# class S_Child(Parent, Child):
#     pass


# 多继承
# class SubChild(Parent, Child):
#     pass

class T_Parent:
    def myMethod(self):
        print("调用父类方法")


class T_Child:
    def myMethod(self):
        print("调用子类方法")


# -----------------     子父类 定义     -----------------


# -----------------     运算符重载    -----------------
class Vector:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return "Vector (%d,%d) " % (self.a, self.b)

    def __add__(self, other):
        return Vector(self.a + other.a, self.b + other.b)


# -----------------     运算符重载    -----------------

# -----------------     类属性与方法    -----------------
class JustCounter:
    # __private_attrs：两个下划线开头，声明该属性为私有，不能在类的外部被使用或直接访问。在类内部的方法中使用时
    # self.__private_attrs。
    # 私有变量
    __secretCount = 0

    # 公开变量
    publicCount = 0

    def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print(self.__secretCount)
# -----------------     类属性与方法    -----------------
