# -*- coding:UTF-8 -*-
'''
Python面向对象
'''

import log_utils as log

# 类的导入
from common_obj import Point, Child, T_Child, Vector, JustCounter

log.loge("类定义与调用")


# 创建类
class Employee:
    '所有员工基类'
    empCount = 0

    def __init__(self, name, salary):
        print("构造方法被调用了")
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print("displayCount方法被调用了")
        print("Total Employee %d " % Employee.empCount)

    def displayEmployee(self):
        print("displayEmployee方法被调用了")
        print("Name: ", self.name, " , Salary: ", self.salary)


"""
empCount 变量是一个类变量，它的值将在这个类的所有实例之间共享。你可以在内部类或外部类使用 Employee.empCount 访问。
第一种方法__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法，当创建了这个类的实例时就会调用该方法
self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
"""

# 创建实例对象
# 实例化类其他编程语言中一般用关键字 new，但是在 Python 中并没有这个关键字，类的实例化类似函数调用方式。
# 以下使用类的名称 Employee 来实例化，并通过 __init__ 方法接收参数。
e1 = Employee("zzg", 2000)
# e1.displayCount()
e1.displayEmployee()
print("--------")
e2 = Employee("zzg2", 2100)
# 访问属性
# 您可以使用点号 . 来访问对象的属性。使用如下类的名称访问类变量:
# e2.displayCount()
e2.displayEmployee()
print("Total Employee %d" % Employee.empCount)

log.loge(" 属性添加删除修改 ")
# 可以使用这种方式直接给对象添加属性，不必在类中定义
e1.age = 17
e2.age = 22
print("E1 age %d" % e1.age)
print("E2 age %d" % e2.age)

# 修改属性
e1.age = 25
print("E1 age %d" % e1.age)

# 删除属性
del e1.age

# 如果存在指定属性，则返回True
print(hasattr(e1, "age"))

# 添加属性
setattr(e1, "age", 18)

# 返回指定属性的值
print(getattr(e1, "age"))

# 删除属性
delattr(e1, "age")
log.loge(" 属性添加删除修改 ")
"""
self代表类的实例，而非类
类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是 self。
"""
log.loge("类定义与调用")


# self代表类的实例，而非类
class Test:
    def prt(self):
        print(self)
        print(self.__class__)


t = Test()
t.prt()


class Test2:
    '测试使用的类'

    def prt(runoob):
        print(runoob)
        print(runoob.__class__)


t = Test2()
t.prt()
print()
log.loge('Python内置类属性')
print('Test2.__doc__', Test2.__doc__)
print('Test2.__name__', Test2.__name__)
print('Test2.__module__', Test2.__module__)
print('Test2.__bases__', Test2.__bases__)
print('Test2.__dict__', Test2.__dict__)
log.loge('Python内置类属性')

# 析构函数 __del__ ，__del__在对象销毁的时候被调用，当对象不再被使用时，__del__方法运行
# python对象销毁(垃圾回收)

log.loge("python对象销毁(垃圾回收)")
pt1 = Point()

pt2 = pt1
pt3 = pt1
# 打印对象的id
print(id(pt1), id(pt2), id(pt3))

del pt1
del pt2
del pt3
log.loge("python对象销毁(垃圾回收)")

# -----------------     子父类 定义     -----------------
log.loge("子父类")
c = Child()  # 实例化子类
c.childMethod()  # 调用子类的方法
c.parentMethod()  # 调用父类方法
c.setAttr(200)  # 再次调用父类的方法 - 设置属性值
c.getAttr()  # 再次调用父类的方法 - 获取属性值

# 布尔函数判断一个类是另一个类的子类或者子孙类
# print(issubclass(c, Parent))

# 布尔函数如果obj是Class类的实例对象或者是一个Class子类的实例对象则返回true。
# print(isinstance(c, Parent))

## 多继承调用(运行失败,以后再校准)
log.loge("多继承")
# sub_child = SubChild()
# sub_child.childMethod()
# sub_child.parentMethod()
# sub_child.setAttr(1500)
# sub_child.getAttr()
log.loge("多继承")

log.loge("方法重写")

t_child = T_Child()
t_child.myMethod()

log.loge("方法重写")

log.loge("子父类")
# -----------------     子父类 定义     -----------------

# -----------------     运算符重载    -----------------
v1 = Vector(2, 10)
v2 = Vector(5, -2)
print(v1 + v2)
# -----------------     运算符重载    -----------------

# -----------------     类属性与方法    -----------------
counter = JustCounter()
counter.count()
counter.count()
print("counter.publicCount ", counter.publicCount)

# 私有变量的外部访问
# Python不允许实例化的类访问私有数据，但你可以使用 object._className__attrName（ 对象名._类名__私有属性名 ）访问属性
print("counter._JustCounter__secretCount", counter._JustCounter__secretCount)

# -----------------     类属性与方法    -----------------
