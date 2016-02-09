---
layout: post
title: Constant Propagation & Folding.
excerpt: Constant propagation and folding are generic optimizations which can greatly simplify the intermediate representation. It is the basis for other optimizations like loop unrolling, loop simplification, function inlining etc.
tags: [compiler, constant propagation, constant folding]
modified: 2016-01-17
comments: true
---

# Constant Propagation & Folding:
## Constant Folding:
Before we define constant propagation, let's define constant folding. Constant folding is a step in which you can "fold" instructions when their operands are constants.
In the instruction below we don't have to emit instructions for computing the result of this instruction. This instruction has both the operands as constants and hence can be computed at compile time, thus getting the result ```a = 20```.

~~~python
a = 5 * 4
~~~

## Constant Propagation:
Constant propagation on the other hand means what the name suggests. It propagates the constants. But propagate where ? Again lets look at an example.

~~~python
def foo():
  a = 5 * 4
  b = 3 * a
  c = 9 * b
  d = 10 * c
  return d

def main():
  g = foo()
  return g * 10
~~~

So this is a contrived example but it does demonstrate the idea very well. In the function "foo", 'a' can be constant folded to 20.
Now, if we propagate 'a' to the next instruction ```b = 3 * a```, which means at compile time we replace the value of 'a' with the constant 20. Now again, ```b = 3 * 20```, so 'b' becomes 60. Now, similarly 'c' becomes 540 and then d becomes 5400.

Cool, so now all the code can be replaced with an equivalent instruction. So the simplified instruction sequence would look like this.

~~~python
def foo():
  return 5400

def main():
  g = foo()
  return g * 10
~~~

So are we done yet? Not really. As you can see that the function foo() will always return 5400. So do we really need to generate code for executing a function? No. We can simplify this function further be replacing its caller by it's value.
So ```g = 5400``` and then 'main' can constant fold the next instruction to like this.

~~~python
def main():
  return 54000
~~~

Now this could be further optimized, if it's the main function in python to return 54000 to the stdout. So this demonstrates the idea of Constant Propagation.

The implementation algorithm is also very intuitive as it does the same as the process we described above.

In my project, I am doing constant folding as the IR builder is building the instructions themselves making it very simple and also when I do constant propagation since that reveals opportunities for folding again.

Look at the files, const_progation.py & irbuilder.py for seeing the idea.
