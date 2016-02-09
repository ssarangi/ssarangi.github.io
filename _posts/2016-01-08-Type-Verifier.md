---
layout: post
title: A Simple Type Verifier for Python Functions using Decorators
excerpt: A Simple Type Verifier for Python Functions using Decorators. Type verification is useful in debugging or when code is written to handle only a specific set of types.
tags: [python3, python, decorators, type verifier]
modified: 2016-01-08
comments: true
---

# Introduction:
Often times while debugging or if you come from a typed system world it is useful to have some type verification. What I mean by that is you have written your functions in python assuming they are some particular type, however the caller calls your function with some bizarre arguments which you didn't expect. Instead of informing the caller that the type passed is incorrect, you crash with a bizarre error message which makes no sense.

# Decorators come to the rescue:
This is where python decorators come in handy. A google search comes up with many helpful results but I have listed one which I particularly like.

[Python Decorators in 12 steps](http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/)

# Usage:
So the first question is how would my code look without such type verification.

~~~python
def raise_exception(actual_type, expected_type):
    raise Exception("Expected %s to be of type %s" % (actual_type, expected_type))


def old_style_foo(a, b):
    if not isinstance(a, int):
        raise_exception(type(a), int)

    if not isinstance(b, str):
        raise_exception(type(b), str)

    print("%s: %s" % (b, a))
~~~

As you can see here, as the number of arguments increase, this becomes quite tedious to do this kind of checking.
Now lets see how it would look with the verifier.

~~~python
@verify(a=int, b=str)
def foo(a, b):
    print("%s: %s" % (b, a))
~~~

We could make the type verification more complex by using Union types too for example something like this.

~~~python
@verify(a=U(int, str), b=str)
def foo(a, b):
    print("%s: %s" % (b, a))
~~~

So in this case 'a' could be either an int or a string. So the type verifier would check for all the types before raising an error.

So lets see how the error message would look in the case when we pass in a function call such as below.

~~~python
foo("saa", "test")
~~~

The error message is "Exception: Expected b to be of type: str but received type: <class 'int'>"

So lets see the code for implementing this verify decorator.

~~~python
def verify(func=None, **options):
    if func is not None:
        # We received the function on this call, so we can define
        # and return the inner function
        def inner(*args, **kwargs):
            if len(options) == 0:
                raise Exception("Expected verification arguments")

            func_code = func.__code__
            arg_names = func_code.co_varnames

            for k, v in options.items():
                # Find the key in the original function
                idx = arg_names.index(k)

                if (len(args) > idx):
                    # get the idx'th arg
                    arg = args[idx]
                else:
                    # Find in the keyword args
                    if k in kwargs:
                        arg = kwargs.get(k)

                if isinstance(v, U):
                    # Unroll the types to check for multiple types
                    types_match = False
                    for dtype in v.types:
                        if isinstance(arg, dtype):
                            types_match = True

                    if types_match == False:
                        raise Exception("Expected " + str(k) + " to be of type: " + str(v) + " but received type: " + str(type(arg)))
                elif not isinstance(arg, v):
                    raise Exception("Expected " + str(k) + " to be of type: " + v.__name__ + " but received type: " + str(type(arg)))

            output = func(*args, **kwargs)
            return output

        return inner
    else:
        # We didn't receive the function on this call, so the return value
        # of this call will receive it, and we're getting the options now.
        def partial_inner(func):
            return verify(func, **options)
        return partial_inner
~~~

The structure above is a standard python decorator structure so I won't explain the nested function structure. Instead I will focus on the type system. Python will deprecate a function called inspect.getargspec which gives the function arguments directly. So I had to directly figure out what argument is to be mapped from the function.

~~~python
func_code = func.__code__
arg_names = func_code.co_varnames
~~~

These 2 lines get the arg names as they appear in function. The next step is to iterate over the argument specification received from the verify decorator.

~~~python
@verify(a=U(int, str), b=str)
~~~

has to be matched to

~~~python
foo("saa", 5)
~~~

So the options gives us what 'a' and 'b' types are supposed to be. We iterate over them to figure out which argument in the caller matches them.

~~~python
for k, v in options.items():
    # Find the key in the original function
    idx = arg_names.index(k)

    if (len(args) > idx):
        # get the idx'th arg
        arg = args[idx]
    else:
        # Find in the keyword args
        if k in kwargs:
            arg = kwargs.get(k)
~~~

This code basically achieves this. It maps "saa" to 'a' and '5' to 'b'.
After that the code is pretty straightforward. If its a union type then it iterates over the types and checks each type. If it's not a union type then it simply checks the base type it gets with the arg.

So the last thing remaining in this is to see the Union class definition.

~~~python
class U:
    def __init__(self, *args):
        self.types = args

    def __str__(self):
        return ",".join(self.types)

    __repr__ = __str__
~~~

That's it. The code is not very complex. I haven't tested thoroughly with keyword args and default arguments but it shouldn't be too difficult to extend this or fix anything if you happen to use this code.

[Github Link](https://github.com/ssarangi/python_type_verifier)
