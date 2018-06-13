资料地址:
http://www.runoob.com/django/django-template.html

## 创建第一个项目
使用 django-admin.py 来创建 HelloWorld 项目：

    django-admin.py startproject HelloWorld

最新版的 Django 请使用 django-admin 命令:

    django-admin startproject HelloWorld

### 目录说明：

    HelloWorld: 项目的容器。
    manage.py: 一个实用的命令行工具，可让你以各种方式与该 Django 项目进行交互。
    HelloWorld/__init__.py: 一个空文件，告诉 Python 该目录是一个 Python 包。
    HelloWorld/settings.py: 该 Django 项目的设置/配置。
    HelloWorld/urls.py: 该 Django 项目的 URL 声明; 一份由 Django 驱动的网站"目录"。
    HelloWorld/wsgi.py: 一个 WSGI 兼容的 Web 服务器的入口，以便运行你的项目。

### 服务器启动:
    python manage.py runserver 0.0.0.0:8000

python manage.py runserver 0.0.0.0:8081
### 错误解决:

    Error: [WinError 10013] 以一种访问权限不允许的方式做了一个访问套接字的尝试。
    端口被占用更换一个端口使用即可


## 视图和 URL 配置
在先前创建的 HelloWorld 目录下的 HelloWorld 目录新建一个 view.py 文件

    from django.http import HttpResponse

    def hello(request):
        return HttpResponse("Hello world! ")

接着，绑定 URL 与视图函数。打开 urls.py 文件，删除原来代码，将以下代码复制粘贴到 urls.py 文件中：
HelloWorld/HelloWorld/urls.py 文件代码：

    from django.conf.urls import url

    from . import view

    urlpatterns = [
        url(r'^$', view.hello),
    ]

## url() 函数
Django url() 可以接收四个参数，分别是两个必选参数：regex、view 和两个可选参数：kwargs、name，接下来详细介绍这四个参数。
    regex: 正则表达式，与之匹配的 URL 会执行对应的第二个参数 view。
    view: 用于执行与正则表达式匹配的 URL 请求。
    kwargs: 视图使用的字典类型的参数。
    name: 用来反向获取 URL。

# Django 模板
在上一章节中我们使用 django.http.HttpResponse() 来输出 "Hello World！"。该方式将数据与视图混合在一起，不符合 Django 的 MVC 思想。

## 模板应用实例
我们接着上一章节的项目将在 HelloWorld 目录底下创建 templates 目录并建立 hello.html文件，整个目录结构如下：

## Django 模板标签
if/else 标签
基本语法格式如下：

    {% if condition %}
         ... display
    {% endif %}

或者：

    {% if condition1 %}
       ... display 1
    {% elif condition2 %}
       ... display 2
    {% else %}
       ... display 3
    {% endif %}

根据条件判断是否输出。if/else 支持嵌套。
{% if %} 标签接受 and ， or 或者 not 关键字来对多个变量做判断 ，或者对变量取反（ not )，例如：

    {% if athlete_list and coach_list %}
         athletes 和 coaches 变量都是可用的。
    {% endif %}

## for 标签
{% for %} 允许我们在一个序列上迭代。
与Python的 for 语句的情形类似，循环语法是 for X in Y ，Y是要迭代的序列而X是在每一个特定的循环中使用的变量名称。
每一次循环中，模板系统会渲染在 {% for %} 和 {% endfor %} 之间的所有内容。
例如，给定一个运动员列表 athlete_list 变量，我们可以使用下面的代码来显示这个列表：

    <ul>
    {% for athlete in athlete_list %}
        <li>{{ athlete.name }}</li>
    {% endfor %}
    </ul>

给标签增加一个 reversed 使得该列表被反向迭代：

    {% for athlete in athlete_list reversed %}
    ...
    {% endfor %}

可以嵌套使用 {% for %} 标签：

    {% for athlete in athlete_list %}
        <h1>{{ athlete.name }}</h1>
        <ul>
        {% for sport in athlete.sports_played %}
            <li>{{ sport }}</li>
        {% endfor %}
        </ul>
    {% endfor %}

## ifequal/ifnotequal 标签
{% ifequal %} 标签比较两个值，当他们相等时，显示在 {% ifequal %} 和 {% endifequal %} 之中所有的值。
下面的例子比较两个模板变量 user 和 currentuser :

    {% ifequal user currentuser %}
        <h1>Welcome!</h1>
    {% endifequal %}

和 {% if %} 类似， {% ifequal %} 支持可选的 {% else%} 标签：8

    {% ifequal section 'sitenews' %}
        <h1>Site News</h1>
    {% else %}
        <h1>No News Here</h1>
    {% endifequal %}

## 注释标签
Django 注释使用 {# #}。

    {# 这是一个注释 #}

过滤器
模板过滤器可以在变量被显示前修改它，过滤器使用管道字符，如下所示：

    {{ name|lower }}

{{ name }} 变量被过滤器 lower 处理后，文档大写转换文本为小写。
过滤管道可以被* 套接* ，既是说，一个过滤器管道的输出又可以作为下一个管道的输入：

    {{ my_list|first|upper }}

以上实例将第一个元素并将其转化为大写。
有些过滤器有参数。 过滤器的参数跟随冒号之后并且总是以双引号包含。 例如：

    {{ bio|truncatewords:"30" }}

这个将显示变量 bio 的前30个词。
其他过滤器：

    addslashes : 添加反斜杠到任何反斜杠、单引号或者双引号前面。
    date : 按指定的格式字符串参数格式化 date 或者 datetime 对象，实例：

        {{ pub_date|date:"F j, Y" }}

    length : 返回变量的长度。

## include 标签
{% include %} 标签允许在模板中包含其它的模板的内容。
下面这个例子都包含了 nav.html 模板：

    {% include "nav.html" %}

## 模板继承
模板可以用继承的方式来实现复用。



# jango 模型
Django 对各种数据库提供了很好的支持，包括：PostgreSQL、MySQL、SQLite、Oracle。
Django 为这些数据库提供了统一的调用API。 我们可以根据自己业务需求选择不同的数据库。
MySQL 是 Web 应用中最常用的数据库。本章节我们将以 Mysql 作为实例进行介绍。你可以通过本站的 MySQL 教程 了解更多Mysql的基础知识。

如果你没安装 mysql 驱动，可以执行以下命令安装：
    sudo pip install mysqlclient

## 数据库配置
我们在项目的 settings.py 文件中找到 DATABASES 配置项，将其信息修改为：
因mysql问题暂放