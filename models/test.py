# encoding=utf-8
"""
协程----微小的进程
yield生成器-----生成一个可迭代对象比如list， tuple，dir
1、包含yield的函数，则是一个可迭代对象（list， tuple等）
每次运行到yield即结束，并保留现场
2、生产者、消费者行为；

3、无需立即执行，需要时才执行
"""

a = [1, 2, 3, 4]
for i in a:
    print(i)

def test():
    i = 0
    a = 4
    while i < a:
        """
        0
        1
        2
        3
        """
        x = yield i
        i += 1


t = test()
print(t)  # <generator object test at 0x0000000002541798>
print(t.__next__())  # 生成器的next（）
print(t.__next__())  # 生成器的next（）
print(t.__next__())  # 生成器的next（）
print(t.__next__())  # 生成器的next（）
# print t.next()  #StopIteration

print(type(range(0, 5)))  # <type 'list'>


def test2():
    x = yield "first, and return"
    print("first %s" % x)
    x = yield "second and return%s" % x
    print("second %s" % x)
    x = yield
    print(x)  # None,没有send


t = test2()
print(t.__next__())
print(t.send("try again"))  # 使用send()则x的值为send的参数，未使用send则x为空
print(t.send("the second args"))

# 1 1 2 3 5 8 13
print("==================")


def test3(num):
    if num == 1:
        yield num
    i = 1
    b = 1
    while i < num:
        x = yield i

        i = b + i


for i in test3(13):
    print(i)

"""
求100000之后的100个质数， 使用yield
"""


def is_p(t_int):
    if t_int > 1:
        for i in range(2, t_int):
            if t_int % i == 0:
                return False
        return True
    else:
        return False


def get_primes():
    i = 1
    while True:
        if is_p(i):
            # x = yield i
            yield i
            i += 1
        i += 1


t = get_primes()
for i in range(0, 100):
    print(t.__next__())