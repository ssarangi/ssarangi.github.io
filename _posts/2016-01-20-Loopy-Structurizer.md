---
layout: post
title: Loopy - A Structurizer for generating structured control flow code.
excerpt: A structurizer is used when the IR is unstructured but the code to be generated needs to be structured. A structured control flow represents idioms which are generally present in higher-level languages such as if-else, for-loop, while-loop, do-while etc.
tags: [compiler, structurizer, unstructured control flow, structured control flow, cfg]
modified: 2016-01-20
comments: true
image: /img/blog/spiderjit/nested_if_structurizer_main_page.png
---

# Introduction

{: .center}
![Nested If Else Block](/img/blog/spiderjit/nested_if_cfg.png "Nested If-else CFG")

In this post I will explain an algorithm which I call Loopy. I developed it independently while looking for algorithms for structuring and is similar to Relooper (used in Emscripten) and Hammock graph based algorithms. However, my intention with this post is to simplify the algorithm and not speak of jargons generally associated with describing such algorithms.

# Some Definitions:

Before I jump into the algorithm, I will just give a brief explanation of the two terms about control flow graphs.

- Structured Control Flow: Control Flow graph which uses if-else, for-loop, while-loop and do-while in the intermediate representation is called structured control flow.

- Unstructured Control Flow: Look at the picture above. It's the basic blocks linked to one another using branch instructions. There is no structure really. However, when we do if-else or our favorite loops (if-else, for, while, do-while), there is a structure since there are no branches which can go haywire.

# Algorithm:

Before I proceed, I describe a couple of classes which I have defined for creating the structurizer. There are 2 classes which will be used for representing the structure of the Control Flow.

~~~python
class Root:
    def __init__(self, parent):
        self.processed = False
        self.__parent = parent

        if parent is not None:
            parent.next = self

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, p):
        self.__parent = p
        if p is not None:
            p.next = self


class ControlFlowBlock(Root):
    def __init__(self, parent = None):
        Root.__init__(self, parent)
        self.true_block = None
        self.false_block = None
        self.cmp_inst = None
        self.nested = None

    def __str__(self):
        s = "CFB: True: %s <--> False: %s" % (self.true_block, self.false_block)
        return s

class Nested(Root):
    def __init__(self, bb, parent=None, next=None):
        Root.__init__(self, parent)
        self.bb = bb
        self.next = next

    def __str__(self):
        s = "Nested: %s" % self.bb
        return s
~~~

The Nested class is used to store a nested block. The idea is that nested blocks link to each other like a linked list, have a link to their parent and create a nesting kind of structure.

Each Control Flow Block is used to recreate the structure and use the true & false blocks to represent the control flow structure.

{: .center}
![Nested If Else Structurized diagram](/img/blog/spiderjit/nested_if_structurizer.png "Nested If Structurized")

The interesting blocks here are the CFB's (Control Flow Block's). The CFB's encompass all the true blocks and false blocks. So the code generation becomes simpler since now the control flow block with keep all the control flow information within it.

The algorithm is simple.

- Create Nested Blocks for every basic block
- Create a Control Flow Block for every block which has a Conditional Branch.
- Do a breadth first search starting with the function's entry block.
- Associate the nested blocks for the true block and false block if a basic block has a conditional branch terminator.
- If the terminator is a branch terminator then just add it to it's parent list.
- If we visit a block which has been visited before during the bfs, we have hit a convergence point.
- Once we get a convergence point, go to the nearest ancestor which is a Control Flow Block and which has not been PROCESSED.
- Set the next block for the CFB as the convergence point.
- Set the CFB as been PROCESSED.
