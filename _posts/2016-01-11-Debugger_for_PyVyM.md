---
layout: post
title: Debugger for PyVyM
excerpt: PyVyM is a Virtual Machine for python and this post describes a debugger for that VM.
tags: [python3, debugger, virtualmachine, vm]
modified: 2016-01-11
comments: true
image: /img/blog/pyvym/debugger.gif
---
# Introduction:
While writing the PyVyM virtual machine, I decided to write a debugger. This might not be the most sophisticated debugger but it does demonstrate the ideas needed to write this.

gdb, lldb are examples of more complicated debuggers used for debugging executable code. However, for the purposes of our VM, its pretty simple to write a debugger.

# Soul of the debugger:

The soul of the debugger is the ability of the debugger to cause the underlying machine to stop whenever it hits it's breakpoints. At that point it should be able to introspect the various states of the machine. For example, registers, memory, threads running etc etc.

However, our VM is a single threaded VM. Very simple indeed. So the main thing needed for this debugger is the ability to set breakpoints and stop at those. It took me a couple of hours to get this to work and I was decently pleased with it considering how simple it was.

# Abilities of the debugger:
- next - Execute Next Instruction
- run - Run VM
-	set bp <loc> - Set Breakpoint at loc
-	disable bp <loc> - Disable Breakpoint at loc
-	clear bp <loc> - Disable Breakpoint at loc
-	clear all bps - Clear all Breakpoints
-	view source <loc> - View Source. If no loc is specified entire source is shown
-	view locals - View the Local variables
-	view globals - View the Global variables
-	view local <var> - View local var
-	view global <var> - View global var
-	view backtrace - View the BackTrace
-	view bp - View Breakpoints
-	help - Display this help
-	quit - Quit

# Implementation:

The main logic of the debugger is as follows:
- Ability to translate byte code offsets to line numbers
- Setup a list of breakpoints (each breakpoint is a line number).
- Run an execute loop which would execute each instruction and then check if any of the breakpoint has been hit.
- Since one python line source could correspond to multiple lines of the bytecode, so executing the <b>next</b> instruction should be able to execute all lines of the bytecode corresponding to that line.

# Line Number mapping:

Line number mapping is the of the two main things needed to implement this. The file debugger_support.py contains a class called LineNo which translates this.

Python code object contains a field called co_lnotab which contains the mapping of the bytecode range and the source line number.

~~~python
class LineNo:
    """
    Described in http://svn.python.org/projects/python/trunk/Objects/lnotab_notes.txt
    """
    def __init__(self, start_lineno, co_lnotab, source, filename):
        self.__start_lineno = start_lineno
        self.__byte_increments = list(co_lnotab[0::2])
        self.__line_increments = list(co_lnotab[1::2])
        self.__source = source
        self.__filename = filename
        self.__currently_executed_line = None

    def line_number(self, ip):
        lineno = addr = 0

        for addr_incr, line_incr in zip(self.__byte_increments, self.__line_increments):
            addr += addr_incr
            if addr > ip:
                return lineno + self.__start_lineno

            lineno += line_incr

        return lineno + self.__start_lineno
~~~

This function described above gives the line numbers. The byte code offsets are not absolute but rather in increments. The link here (http://svn.python.org/projects/python/trunk/Objects/lnotab_notes.txt) gives the details of how the code is presented.

So we keep on adding increments to see if we have exceeded the ip. Once the address exceeds the current ip we get the next line number mapped to the bytecode.

# Overriding the execute loop:

The VM has an execute loop which runs through the code. So the debugger has to make sure that it runs its own execute loop. The VM exposes interfaces to run an individual instruction so it's easy for the debugger to run each instruction and then check all the breakpoints set.

However, problems arise when function calls are made. Function calls are typically implemented with its own stack frame and once a new stack frame is created, the call function calls the vm's execute method. So let's see why this is a problem.

~~~python
def foo(a, b):
  return a + b

foo(3, 4)
~~~

So the debugger starts with setting up the module, loading the arguments into the stack. Till this point it is fine since the debugger is running its own execute method which checks for breakpoints.

Now once it sees the function call foo(3, 4), it has to invoke the function foo's code. At this point, the VM calls its own execute method which is a loop to run all the instructions in the foo method. This means that the debugger has no way of checking after every instruction if any breakpoint has been set.

# Debugger Hookup:

So the way to hook up the debugger to the vm is for the debugger to override the vm's execute method. This is very simple. When we instantiate the vm's object we just override the execute method with the debugger's execute method.

~~~python
def initialize_vm(self, code, source, filename):
    self.__vm = BytecodeVM(code, source, filename)
    config = VMConfig()
    self.__vm.config = config
    config.show_disassembly = True
    self.__vm.execute = self.execute
~~~

This snippet of code shows how the execute method is overridden. The last line does it. Once this is done, the only problem remaining is that the debugger's execute method has a loop which displays a prompt.

~~~python
def execute(self, call_from_vm = True):
    while True:
        arg1 = None

        if not call_from_vm:
            cmd_res = self.display_prompt()
            if isinstance(cmd_res, tuple):
                cmd = cmd_res[0]
                arg1 = cmd_res[1]
            else:
                cmd = cmd_res
        else:
            cmd = DebuggerCmds.VM_RUN
~~~

This snippet shows how the prompt is to be displayed. So when the VM CALL_FUNCTION calls the execute method it comes into the debugger's execute method with the "call_from_vm" parameter "True". This indicates that this call is to be run continuously without displaying the prompt till we hit a breakpoint.

# Viewing & Setting Locals:

Viewing and setting locals is similar to the how the VM internally does it. It goes up the exec_frame stack to find out if any of the exec_frames contains the local we are interested in. If so it set's it or returns it. The only thing to take into account here is that the type of the variable is unknown and the value to set from the command prompt comes in the form of a string. So the solution to this is to check for the existing type of the variable. If we can typecast the set value to that type then we continue otherwise we set it as a string itself.

# Debugger in action:

{: .center}
![Debugger](/img/blog/pyvym/debugger.gif "Debugger")