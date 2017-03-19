---
layout: post
title: Adapting Luajit for Python 3
excerpt: I have looked at spidermonkey and a few other JIT engines and kept on reading about LuaJit over and over again. I finally decided to look at Luajit closely and realized that it is a pretty complex piece of engineering but done single-handedly by Mike Pall. The number of source files is way smaller compared to spidermonkey or v8 but that's probably because Lua is much simpler than javascript. Since is fairly closer to python I wanted to try my hands out to modify Luajit's code to experiment if I could run basic python on it.
tags: [jit, lua, luajit, python]
modified: 2016-02-04
comments: true
crosspost_to_medium: true
---

# Luajit is for Lua !!!

I know... But out of curiosity and wanting to see more of it I decided to go ahead and try and see what I could get Luajit to run.

# Luajit Architecture:

Luajit is written completely in C and has a huge helping hand from assembly too. But overall if you look at it, it has the same number of things a regular jitter does.

- It lexes and parses lua code
- Emit's a bytecode from it.
- From the bytecode it emits an SSA IR
- It then optimizes the SSA IR.
- It also has an interpreter which acts as a tracing jit.
- Finally after identifying hotspots, it runs it on the full jit.
- Garbage collection is also present and Mike had ideas on the next gen garbage collectors too.

# Information:
Look on the Luajit Wiki. It has tons of information.

- [Luajit trace compiler](https://news.ycombinator.com/item?id=6819702)
- [Luajit Dynasm](http://blog.reverberate.org/2012/12/hello-jit-world-joy-of-simple-jits.html)
- [Optimization links](http://stackoverflow.com/questions/7167566/luajit-2-optimization-guide)
- [Peeking inside Luajit](https://pwparchive.wordpress.com/2012/10/16/peeking-inside-luajit/)

# Code which is working:
[Github Link](https://github.com/ssarangi/luajit)

- Simple functions
- lists
- While loops
- If-else-elseif control flow

There is a hell lot of work left and am not even sure if everything can be / should be supported by this. If you would like to collaborate on this, drop me a line.

~~~python
def foo():
    print("hello world")
    a = 3
    b = 2
    c = a * b
    return c

g = foo()
print(g)
~~~

~~~python
def foo():
    a = 3
    b = 3
    if a > b:
        print("a > b")
    elif a == b:
        print("a == b")
    else:
        print("b > a")

foo()
~~~

~~~python
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
print(days[0])
i = 3
i = i + 1
print(days[i])
print(days[4])
~~~

~~~python
def foo():
    a = 1
    while a < 10:
        print(a)
        a = a + 1

foo()
~~~