#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample02_线程.py
# @Date    :   2018/8/8
# @Desc    :

import time


def print_time(threadName, delay):
    count = 0
    while count < 5:
        time.sleep(delay)
        count += 1
        print("%s:%s" % (threadName, time.ctime(time.time())))


import _thread
import threading

"""
使用_thread创建线程 
"""


def test01():
    try:
        # Python中使用线程有两种方式：函数或者用类来包装线程对象。
        _thread.start_new_thread(print_time, ("Thread-1", 2,))
        _thread.start_new_thread(print_time, ("Thread-2", 4,))
    except:
        print("Error:无法启动线程")

    while 1: pass
    pass


exitFlag = 0


def print_time2(threadName, delay, counter):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1
    pass


class myThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("开始线程:" + self.name)
        print_time2(self.name, self.counter, 5)
        print("退出线程:" + self.name)


"""
使用 threading 模块创建线程
"""


def test02():
    thread1 = myThread(1, "Thread-1", 1)
    thread2 = myThread(2, "Thread-2", 2)

    # 开启线程
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    print("退出主线程")
    pass


threadLock = threading.Lock()


# 同步线程
def print_time3(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1


class mySyncThread(threading.Thread):

    def __init__(self, threadIdD, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadIdD
        self.name = name
        self.counter = counter

    def run(self):
        print("开启线程: " + self.name)
        # 获取锁，用于线程同步
        threadLock.acquire()
        print_time3(self.name, self.counter, 3)
        # 释放锁，开启下一个线程
        threadLock.release()


def test3():
    threads = []
    # 创建新线程
    thread1 = myThread(1, "Thread-1", 1)
    thread2 = myThread(2, "Thread-2", 2)

    # 开启新线程
    thread1.start()
    thread2.start()

    # 添加线程到线程列表
    threads.append(thread1)
    threads.append(thread2)

    # 等待所有线程完成
    for t in threads:
        t.join()
    print("退出主线程")
    pass



if __name__ == '__main__':
    # test01()
    # test02()
    test3()
