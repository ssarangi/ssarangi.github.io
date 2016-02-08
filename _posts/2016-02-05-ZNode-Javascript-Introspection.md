---
layout: post
title: ZNode - A Javascript Introspection module
excerpt: ZNode is a minimalistic browser based flowchart module. I took this project and modified it heavily so that it could do some basic javascript introspection and show them as blocks. Think of it like UML diagram generation.
tags: [javascript, introspection, znode]
modified: 2016-02-04
---
[Github](https://github.com/ssarangi/JsIntrospection)

# Introduction:
This was a pretty old project of mine which I just found laying around. Thinking that this could become a good post just to introduce how basic introspection works. Of course as usual, this was my first foray into the compiler land and javascript is a beast. As such this also happened to be my first project using javascript.

# Background
ZNode is a pretty neat project which lets you draw flowcharts online. The interface is pretty nice and it uses raphael.js to do the diagrams. (http://zreference.com/projects/Znode/src/)
So while playing around with javascript I decided to give this is a shot and see if I could get it to generate diagrams and information for the code we write.

# Architecture:
My first disclaimer is that I didn't write the javascript parser. I borrowed the parser and then started writing code to link the various information together. As of now I don't remember anything from the code to explain it but if you are curious enough, then drop me a line and I can look back at the code. :)

So basic idea is
- Select a file or paste some code
- The code is parsed first using the parser
- The introspection module then kicks in and creates all the hierarchy trees.
- Once the introspection is done, the diagrams are spit out to display the information succinctly.

# Example:
~~~javascript
// Declare two objects - we're going to want Lion to
// inherit from cat
function cat()
{
  this.eyes = 2;
  this.legs = 4;
  this.diet = 'carnivore';

  this.catFunction = function() { return 5; }

  return true;
}

function lion()
{
  this.mane = true;
  this.origin = 'Africa';

  this.catFunction();

  return true;
}

// Now comes the inheritance
lion.prototype = new cat();

// We can now obtain lion.diet
var simba = new lion();
var simba_diet = simba.diet;
~~~

# Output:

{: .center}
![Diagram showing inheritance](/img/blog/znode/simple_inheritance_1.png "Overall Inheritance")

As can be seen, lion derives from cat. So this diagram illustrates that. See the arrow from lion to the cat showing the inheritance. The rest simba and simba_diet are global variables.

Now look at the box cat. Clicking on the button "F" shows all the functions. Clicking on the function shows its uses.

{: .center}
![Pressing F on cat](/img/blog/znode/simple_inheritance_2.png "Clicking on button F for cat")
![clicking on the function](/img/blog/znode/simple_inheritance_3.png "Clicking on function for cat")

{: .center}
![Data Members for cat](/img/blog/znode/simple_inheritance_4.png "Clicking on function for cat")

This shows some of the straightforward usage on a simple code.

{: .center}
![Demo](/img/blog/znode/znode_demo.gif "Demo")
