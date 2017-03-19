---
layout: post
title: Compiler IR - Considerations and Implementation
excerpt: A Compiler intermediate representation - consideration and implementation details.
tags: [compiler, intermediate representation, llvm ir]
modified: 2016-01-16
comments: true
crosspost_to_medium: true
---

# Compiler IR:

While working on my project for a jit in python, one of the main design considerations was to select an appropriate IR. An IR or Intermediate Representation of an input source program. So what's the use of an IR ? Why the hell do I need one to begin with?

So let's say you decided to write a compiler for some esoteric language you devised or even python / ruby for that matter. Let's say your goal is to generate javascript code out of it. So you implemented a lexer and parser for the language, or better still, if you are writing a compiler for python, you used the module "ast" to build an Abstract Syntax Tree for it and then you can directly do a source to source translation to javascript from the ast itself. Infact, PythonJS(https://github.com/PythonJS/PythonJS) does exactly this.

So you might be wondering at this point, what else is needed? Where does the whole IR concept fit into this?

# Why an IR?

So let's continue the hypothetical situation we described above. You generated javascript code but realized it is not optimal. So you write some optimizations on the AST. All fine and dandy indeed !!!

Now you decide to write a compiler for another language but you find out that you need another AST and for generating efficient code you need to rewrite all your optimizations. So this is where an Intermediate Representation (IR) is useful.

An IR is just a mid-level language to which your compiler frontends convert to. An IR could be either very close to the original source language or be very close to the target representation and all these depend on your requirements.

For example, LLVM has 2 levels or IR. A regular IR which is relatively high level and supports a lot of constructs like switch, functions, modules etc yet lower level than C / C++. The other level of IR is called MachineIR which is very close to current hardware targets like x86, SPARC, Mips etc. LLVM does a lot of optimizations at it's higher level IR before translating to a lower level IR.

# IR - Design Considerations:

So what are the things to consider when designing an IR. An IR changes the direction in which one has to write optimizations and hence poses a quite an important decision.

- SSA vs Non-SSA
- Unstructured vs Structured Control Flow

These are the 2 main considerations I can think of while designing an IR. The Dragon book for example prescribes a TAC (Three Address Code) format of representing IR. The IR looks very low level and very close to x86 assembly. However, it is Non-SSA and uses a Structured Control Flow approach.

I am more familiar with SSA based IR from LLVM which also happens to be Unstructured Control Flow.

So what is the difference between Structured and Unstructured control Flow?
- Structured Control flow uses the higher level language constructs like if-then-else-endif, while, do-while, for etc.
- Unstructured on the other hand can only use branch instructions, conditional branch instructions (goto kind of instructions) and labels to implement the higher level control flow.
- Typed or Typeless

So some of the advantages of SSA are that def-use chains are easy to build since every instruction is a def and all values are unique. Also a variable cannot have more than one def making it easy to do optimizations like Dead Code Elimination.

However, on the other hand, it introduces the concept of phi-nodes which means there have to be optimizations which convert to a phi-based ir and then during the code generation phase have to convert it out of a phi-based ir to the target code. These are not relatively straightforward to do.

Now with unstructured control flow, its easy to build the control flow graph and do analysis passes like DominatorTree pass etc but it makes it slightly more complicated to do optimizations on loops (since loops have to be recreated, induction variables recalculated etc) as well as generating code for a structured control flow target. For example, from python to javascript, if we go through an unstructued control flow based IR then we have to write a structurizer which will get back the original structure from the IR to generate code since javascript doesn't support goto's and this means that fairly complex and inefficient code will be generated in this case. Needing a structurizer also means that in some cases it might be very difficult to recreate the structure if the mid-end optimizations are not careful.

Typed or Typeless discussions are important based on the language from which it is being converted. For example for languages which are statically typed (like C/C++/C#,Java etc) it makes sense to use a typed IR. However, for dynamic languages like python, ruby etc using a typed IR has its own challenges. You need some kind of type inference to try and predict the types or you need to create complex objects which would need boxing and unboxing when working with different types.

# IR Implementation:

So I had worked with LLVM before and was fairly familiar with it's IR when I started working on this project. So my first obvious choice would be have been to integrate LLVM using some python bindings. However, most of the python bindings seem to be have died since people couldn't keep up with maintaining with llvm's pace (or maybe they got bored ;) ). The only stable version which seems to be there is llvmlite from numba project. Looking at llvmlite, I decided to try and write my own IR which would be very similar to llvm ir but would not require me to link to llvm libraries.

The code is not very complex for designing the IR, but for the first step, I decided to make it typeless so that I don't have to write a type inferer immediately. This made it relatively easier to do the translation from the frontend to the midend. However, it does make doing optimizations more difficult, specially alias analysis etc. However, I think I can live without that for now.

~~~python
def main(a, b, c):
    d = 0
    if a > b:
        if b > c:
            c = b
            a = b
            d = a * c
        else:
            c = a * b
            a = c - b
            d = a - c
        d = d * 5
    else:
        d = b

    return d
~~~

For this python source above, my current IR being generated is shown below.

~~~
Module: root_module
define  main(a, b, c) {
<entry>:
    %0 = icmp sle a, b
    br %0 1, label %then, label %else

<then>: ; pred: ['entry']
    %1 = icmp sle b, c
    br %1 1, label %then_1, label %else_1

<then_1>: ; pred: ['then']
    %mult = mul(b, b)
    br endif_1

<else_1>: ; pred: ['then']
    %mult1 = mul(b, b)
    %sub = sub(%mult1, b)
    %sub1 = sub(%sub, %mult1)
    br endif_1

<endif_1>: ; pred: ['else_1', 'then_1']
    %mult2 = mul(%sub1, 5)
    br endif

<else>: ; pred: ['entry']
    br endif

<endif>: ; pred: ['else', 'endif_1']
    return b
}
~~~

The IR looks very similar to llvm (except for its printing aspects). The output below shows how the control flow graph looks for this particular IR.

{: .center}
![CFG](/img/blog/spiderjit/nested_if_cfg.png "Control Flow Graph")

In future posts, I will talk more about some of the optimizations I am doing with this IR.

[Github Link](https://github.com/ssarangi/spiderjit)
