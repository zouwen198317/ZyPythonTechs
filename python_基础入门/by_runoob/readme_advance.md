# Python 网络编程
Python 提供了两个级别访问的网络服务。：

    低级别的网络服务支持基本的 Socket，它提供了标准的 BSD Sockets API，可以访问底层操作系统Socket接口的全部方法。
    高级别的网络服务模块 SocketServer， 它提供了服务器中心类，可以简化网络服务器的开发。

## 什么是 Socket?
Socket又称"套接字"，应用程序通常通过"套接字"向网络发出请求或者应答网络请求，使主机间或者一台计算机上的进程间可以通讯。

##socket()函数
Python 中，我们用 socket（）函数来创建套接字，语法格式如下：

    socket.socket([family[, type[, proto]]])

## 参数

    family: 套接字家族可以使AF_UNIX或者AF_INET
    type: 套接字类型可以根据是面向连接的还是非连接分为SOCK_STREAM或SOCK_DGRAM
    protocol: 一般不填默认为0.

## Socket 对象(内建)方法

    函数	描述
    服务器端套接字
    s.bind()	绑定地址（host,port）到套接字， 在AF_INET下,以元组（host,port）的形式表示地址。
    s.listen()	开始TCP监听。backlog指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
    s.accept()	被动接受TCP客户端连接,(阻塞式)等待连接的到来
    客户端套接字
    s.connect()	主动初始化TCP服务器连接，。一般address的格式为元组（hostname,port），如果连接出错，返回socket.error错误。
    s.connect_ex()	connect()函数的扩展版本,出错时返回出错码,而不是抛出异常
    公共用途的套接字函数
    s.recv()	接收TCP数据，数据以字符串形式返回，bufsize指定要接收的最大数据量。flag提供有关消息的其他信息，通常可以忽略。
    s.send()	发送TCP数据，将string中的数据发送到连接的套接字。返回值是要发送的字节数量，该数量可能小于string的字节大小。
    s.sendall()	完整发送TCP数据，完整发送TCP数据。将string中的数据发送到连接的套接字，但在返回之前会尝试发送所有数据。成功返回None，失败则抛出异常。
    s.recvfrom()	接收UDP数据，与recv()类似，但返回值是（data,address）。其中data是包含接收数据的字符串，address是发送数据的套接字地址。
    s.sendto()	发送UDP数据，将数据发送到套接字，address是形式为（ipaddr，port）的元组，指定远程地址。返回值是发送的字节数。
    s.close()	关闭套接字
    s.getpeername()	返回连接套接字的远程地址。返回值通常是元组（ipaddr,port）。
    s.getsockname()	返回套接字自己的地址。通常是一个元组(ipaddr,port)
    s.setsockopt(level,optname,value)	设置给定套接字选项的值。
    s.getsockopt(level,optname[.buflen])	返回套接字选项的值。
    s.settimeout(timeout)	设置套接字操作的超时期，timeout是一个浮点数，单位是秒。值为None表示没有超时期。一般，超时期应该在刚创建套接字时设置，因为它们可能用于连接的操作（如connect()）
    s.gettimeout()	返回当前超时期的值，单位是秒，如果没有设置超时期，则返回None。
    s.fileno()	返回套接字的文件描述符。
    s.setblocking(flag)	如果flag为0，则将套接字设为非阻塞模式，否则将套接字设为阻塞模式（默认值）。非阻塞模式下，如果调用recv()没有发现任何数据，或send()调用无法立即发送数据，那么将引起socket.error异常。
    s.makefile()	创建一个与该套接字相关连的文件

## Python Internet 模块
以下列出了 Python 网络编程的一些重要模块：

    协议	功能用处	端口号	Python 模块
    HTTP	网页访问	80	httplib, urllib, xmlrpclib
    NNTP	阅读和张贴新闻文章，俗称为"帖子"	119	nntplib
    FTP	文件传输	20	ftplib, urllib
    SMTP	发送邮件	25	smtplib
    POP3	接收邮件	110	poplib
    IMAP4	获取邮件	143	imaplib
    Telnet	命令行	23	telnetlib
    Gopher	信息查找	70	gopherlib, urllib


#Python SMTP发送邮件
SMTP（Simple Mail Transfer Protocol）即简单邮件传输协议,它是一组用于由源地址到目的地址传送邮件的规则，由它来控制信件的中转方式。
python的smtplib提供了一种很方便的途径发送电子邮件。它对smtp协议进行了简单的封装。
Python创建 SMTP 对象语法如下：

    import smtplib

    smtpObj = smtplib.SMTP( [host [, port [, local_hostname]]] )
参数说明：

    host: SMTP 服务器主机。 你可以指定主机的ip地址或者域名如: runoob.com，这个是可选参数。
    port: 如果你提供了 host 参数, 你需要指定 SMTP 服务使用的端口号，一般情况下 SMTP 端口号为25。
    local_hostname: 如果 SMTP 在你的本机上，你只需要指定服务器地址为 localhost 即可。

Python SMTP 对象使用 sendmail 方法发送邮件，语法如下：

    SMTP.sendmail(from_addr, to_addrs, msg[, mail_options, rcpt_options])

参数说明：

    from_addr: 邮件发送者地址。
    to_addrs: 字符串列表，邮件发送地址。
    msg: 发送消息

这里要注意一下第三个参数，msg 是字符串，表示邮件。我们知道邮件一般由标题，发信人，收件人，邮件内容，附件等构成，发送邮件的时候，
要注意 msg 的格式。这个格式就是 smtp 协议中定义的格式。



# Python3 多线程
多线程类似于同时执行多个不同程序，多线程运行有如下优点：

    使用线程可以把占据长时间的程序中的任务放到后台去处理。
    用户界面可以更加吸引人，这样比如用户点击了一个按钮去触发某些事件的处理，可以弹出一个进度条来显示处理的进度
    程序的运行速度可能加快
    在一些等待的任务实现上如用户输入、文件读写和网络收发数据等，线程就比较有用了。在这种情况下我们可以释放一些珍贵的资源如内存占用等等。

线程在执行过程中与进程还是有区别的。每个独立的线程有一个程序运行的入口、顺序执行序列和程序的出口。但是线程不能够独立执行，必须依存在应用程序中，由应用程序提供多个线程执行控制。
每个线程都有他自己的一组CPU寄存器，称为线程的上下文，该上下文反映了线程上次运行该线程的CPU寄存器的状态。
指令指针和堆栈指针寄存器是线程上下文中两个最重要的寄存器，线程总是在进程得到上下文中运行的，这些地址都用于标志拥有线程的进程地址空间中的内存。

    线程可以被抢占（中断）。
    在其他线程正在运行时，线程可以暂时搁置（也称为睡眠） -- 这就是线程的退让。

线程可以分为:

    内核线程：由操作系统内核创建和撤销。
    用户线程：不需要内核支持而在用户程序中实现的线程。

Python3 线程中常用的两个模块为：
    _thread
    threading(推荐使用)

thread 模块已被废弃。用户可以使用 threading 模块代替。所以，在 Python3 中不能再使用"thread" 模块。为了兼容性，Python3 将 thread 重命名为 "_thread"。

## 开始学习Python线程
Python中使用线程有两种方式：函数或者用类来包装线程对象。
函数式：调用 _thread 模块中的start_new_thread()函数来产生新线程。语法如下:

    _thread.start_new_thread ( function, args[, kwargs] )

参数说明:

    function - 线程函数。
    args - 传递给线程函数的参数,他必须是个tuple类型。
    kwargs - 可选参数。

## 线程模块
Python3 通过两个标准库 _thread 和 threading 提供对线程的支持。
_thread 提供了低级别的、原始的线程以及一个简单的锁，它相比于 threading 模块的功能还是比较有限的。
threading 模块除了包含 _thread 模块中的所有方法外，还提供的其他方法：

    threading.currentThread(): 返回当前的线程变量。
    threading.enumerate(): 返回一个包含正在运行的线程的list。正在运行指线程启动后、结束前，不包括启动前和终止后的线程。
    threading.activeCount(): 返回正在运行的线程数量，与len(threading.enumerate())有相同的结果。

除了使用方法外，线程模块同样提供了Thread类来处理线程，Thread类提供了以下方法:

    run(): 用以表示线程活动的方法。
    start():启动线程活动。
    join([time]): 等待至线程中止。这阻塞调用线程直至线程的join() 方法被调用中止-正常退出或者抛出未处理的异常-或者是可选的超时发生。
    isAlive(): 返回线程是否活动的。
    getName(): 返回线程名。
    setName(): 设置线程名。

## 使用 threading 模块创建线程
我们可以通过直接从 threading.Thread 继承创建一个新的子类，并实例化后调用 start() 方法启动新线程，即它调用了线程的 run() 方法：


## 线程同步
如果多个线程共同对某个数据修改，则可能出现不可预料的结果，为了保证数据的正确性，需要对多个线程进行同步。
使用 Thread 对象的 Lock 和 Rlock 可以实现简单的线程同步，这两个对象都有 acquire 方法和 release 方法，对于那些需要每次只允许一个
线程操作的数据，可以将其操作放到 acquire 和 release 方法之间。如下：

    多线程的优势在于可以同时运行多个任务（至少感觉起来是这样）。但是当线程需要共享数据时，可能存在数据不同步的问题。
    考虑这样一种情况：一个列表里所有元素都是0，线程"set"从后向前把所有元素改成1，而线程"print"负责从前往后读取列表并打印。
    那么，可能线程"set"开始改的时候，线程"print"便来打印列表了，输出就成了一半0一半1，这就是数据的不同步。为了避免这种情况，引入了锁的
    概念。
    锁有两种状态——锁定和未锁定。每当一个线程比如"set"要访问共享数据时，必须先获得锁定；如果已经有别的线程比如"print"获得锁定了，那么就
    让线程"set"暂停，也就是同步阻塞；等到线程"print"访问完毕，释放锁以后，再让线程"set"继续。

经过这样的处理，打印列表时要么全部输出0，要么全部输出1，不会再出现一半0一半1的尴尬场面。


## 线程优先级队列（ Queue）
Python 的 Queue 模块中提供了同步的、线程安全的队列类，包括FIFO（先入先出)队列Queue，LIFO（后入先出）队列LifoQueue，和优先级队列 PriorityQueue。
这些队列都实现了锁原语，能够在多线程中直接使用，可以使用队列来实现线程间的同步。
Queue 模块中的常用方法:
    Queue.qsize() 返回队列的大小
    Queue.empty() 如果队列为空，返回True,反之False
    Queue.full() 如果队列满了，返回True,反之False
    Queue.full 与 maxsize 大小对应
    Queue.get([block[, timeout]])获取队列，timeout等待时间
    Queue.get_nowait() 相当Queue.get(False)
    Queue.put(item) 写入队列，timeout等待时间
    Queue.put_nowait(item) 相当Queue.put(item, False)
    Queue.task_done() 在完成一项工作之后，Queue.task_done()函数向任务已经完成的队列发送一个信号
    Queue.join() 实际上意味着等到队列为空，再执行别的操作

# Python3 XML解析

# 什么是XML？
XML 指可扩展标记语言（eXtensible Markup Language），标准通用标记语言的子集，是一种用于标记电子文件使其具有结构性的标记语言。 你可以通过本站学习XML教程
XML 被设计用来传输和存储数据。
XML是一套定义语义标记的规则，这些标记将文档分成许多部件并对这些部件加以标识。
它也是元标记语言，即定义了用于定义其他与特定领域有关的、语义的、结构化的标记语言的句法语言。

# python对XML的解析
常见的XML编程接口有DOM和SAX，这两种接口处理XML文件的方式不同，当然使用场合也不同。
python有三种方法解析XML，SAX，DOM，以及ElementTree:

## 1.SAX (simple API for XML )
python 标准库包含SAX解析器，SAX用事件驱动模型，通过在解析XML的过程中触发一个个的事件并调用用户定义的回调函数来处理XML文件。

## 2.DOM(Document Object Model)
将XML数据在内存中解析成一个树，通过对树的操作来操作XML。

## python使用SAX解析xml
SAX是一种基于事件驱动的API。
利用SAX解析XML文档牵涉到两个部分:解析器和事件处理器。
解析器负责读取XML文档,并向事件处理器发送事件,如元素开始跟元素结束事件;
而事件处理器则负责对事件作出相应,对传递的XML数据进行处理。

    1、对大型文件进行处理；
    2、只需要文件的部分内容，或者只需从文件中得到特定信息。
    3、想建立自己的对象模型的时候。

在python中使用sax方式处理xml要先引入xml.sax中的parse函数，还有xml.sax.handler中的ContentHandler。

## ContentHandler类方法介绍

characters(content)方法
调用时机：
从行开始，遇到标签之前，存在字符，content的值为这些字符串。
从一个标签，遇到下一个标签之前， 存在字符，content的值为这些字符串。
从一个标签，遇到行结束符之前，存在字符，content的值为这些字符串。
标签可以是开始标签，也可以是结束标签。
### startDocument()方法
文档启动的时候调用。
### endDocument()方法
解析器到达文档结尾时调用。
### startElement(name, attrs)方法
遇到XML开始标签时调用，name是标签的名字，attrs是标签的属性值字典。
### endElement(name)方法
遇到XML结束标签时调用。

## make_parser方法
以下方法创建一个新的解析器对象并返回。

    xml.sax.make_parser( [parser_list] )

参数说明:

    parser_list - 可选参数，解析器列表

## parser方法
以下方法创建一个 SAX 解析器并解析xml文档：

    xml.sax.parse( xmlfile, contenthandler[, errorhandler])

参数说明:

    xmlfile - xml文件名
    contenthandler - 必须是一个ContentHandler的对象
    errorhandler - 如果指定该参数，errorhandler必须是一个SAX ErrorHandler对象

## parseString方法
parseString方法创建一个XML解析器并解析xml字符串：

    xml.sax.parseString(xmlstring, contenthandler[, errorhandler])

参数说明:

    xmlstring - xml字符串
    contenthandler - 必须是一个ContentHandler的对象
    errorhandler - 如果指定该参数，errorhandler必须是一个SAX ErrorHandler对象


## 使用xml.dom解析xml
文件对象模型（Document Object Model，简称DOM），是W3C组织推荐的处理可扩展置标语言的标准编程接口。
一个 DOM 的解析器在解析一个 XML 文档时，一次性读取整个文档，把文档中所有元素保存在内存中的一个树结构里，之后你可以利用DOM 提供的
不同的函数来读取或修改文档的内容和结构，也可以把修改过的内容写入xml文件。


# Python3 JSON 数据解析
JSON (JavaScript Object Notation) 是一种轻量级的数据交换格式。它基于ECMAScript的一个子集。
Python3 中可以使用 json 模块来对 JSON 数据进行编解码，它包含了两个函数：

    json.dumps(): 对数据进行编码。
    json.loads(): 对数据进行解码。

在json的编解码过程中，python 的原始类型与json类型会相互转换，具体的转化对照如下：

## Python 编码为 JSON 类型转换对应表：

    Python	JSON
    dict	object
    list, tuple	array
    str	string
    int, float, int- & float-derived Enums	number
    True	true
    False	false
    None	null

## JSON 解码为 Python 类型转换对应表：

    JSON	Python
    object	dict
    array	list
    string	str
    number (int)	int
    number (real)	float
    true	True
    false	False
    null	None