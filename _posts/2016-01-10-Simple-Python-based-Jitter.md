---
layout: post
title: A simple python based JIT
excerpt: A simple python based JIT with a C module backend which uses mmap to execute arbitrary x86 code.
tags: [python3, python, jit]
modified: 2016-01-10
comments: true
image: /img/blog/spiderjit/spiderjit.png
---

#Introduction:
JIT or Just-in-Time compilation has been a hot topic in recent years with Jitter's popping up for all kinds of languages. The most prominent one of them is the one for Javascript from both Google and Mozilla known as V8 and SpiderMonkey respectively. Obviously, python followers joined the bandwagon with a bunch of jitter's for python, like pyston from Dropbox, Numba from the Anaconda distribution folks etc.

Jit-ting is fundamental idea in which the compiler generates the x86 assembly code at runtime and executes it. Optimizations like hot-spot analysis etc are done to further decide when to actually JIT and this field is huge and growing.

So this weekend, I was tinkering around with some simple jit blogs when I came across a very interesting blog which is a must read. [Joy of Simple JIT](http://blog.reverberate.org/2012/12/hello-jit-world-joy-of-simple-jits.html)
As I was looking at the code I decided to see if I can make it run via python so that I could write a simple jit for my own language some day. The blog well explains the concepts relating to using mmap with PROT_EXEC permissions so execute assembly. Read the blog before you continue any further.

# First Step

For this to work, the first step is to write a C python module which would expose the mmap functionality to python. Python gives access to mmap features from its internal libraries but they don't give the EXEC permissions and hence I had to go the C route. So the first step was to look up how a C python module would work. For now, this code works only on linux, since it is a pain to get python modules compiling in windows.

So create a new directory which will hold the project. In that create a file named <b>setup.py</b>.

~~~python
from distutils.core import setup, Extension

jittermodule = Extension('jitter',
                         sources = ['jitter/jittermodule.c'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is the Jitter Module for Pyasm2',
       ext_modules = [jittermodule])
~~~

The first line is the important line. It shows the location of the jitter module. According to this setup.py, my C file is located in a folder named <b>jitter</b>

Name this file <b>jittermodule.c</b>

~~~C
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

// Allocates RW memory of given size and returns a pointer to it. On failure,
// prints out the error and returns NULL. mmap is used to allocate, so
// deallocation has to be done with munmap, and the memory is allocated
// on a page boundary so it's suitable for calling mprotect.
void* alloc_writable_memory(size_t size) {
  void* ptr = mmap(0, size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
  if (ptr == (void*)-1) {
    perror("mmap");
    return NULL;
  }
  return ptr;
}

// Sets a RX permission on the given memory, which must be page-aligned. Returns
// 0 on success. On failure, prints out the error and returns -1.
int make_memory_executable(void* m, size_t size) {
  if (mprotect(m, size, PROT_READ | PROT_EXEC) == -1) {
    perror("mprotect");
    return -1;
  }
  return 0;
}

void emit_code_into_memory(unsigned char* m, unsigned char *code) {
  /*
  unsigned char code[] = {
    0x48, 0x89, 0xf8,                   // mov %rdi, %rax
    0x48, 0x83, 0xc0, 0x04,             // add $4, %rax
    0xc3                                // ret
  };
  */
  memcpy(m, code, sizeof(code));
}

const size_t SIZE = 1024;
typedef long (*JittedFunc)(long);

// Allocates RW memory, emits the code into it and sets it to RX before
// executing.
int emit_to_rw_run_from_rx(unsigned char *code) {
  void* m = alloc_writable_memory(SIZE);
  emit_code_into_memory(m, code);
  make_memory_executable(m, SIZE);

  JittedFunc func = m;
  int result = func(4);
  return result;
}

static PyObject *JitterError;

static PyObject* jitter_jit(PyObject *self, PyObject *args) {
    const char* str;
    char * buf;
    Py_ssize_t count;
    PyObject * result;
    int i;

    if (!PyArg_ParseTuple(args, "z#", &str, &count))
    {
        return NULL;
    }

    int buffer_size = (int)count;

    printf("Initiailzed Jitter with code size: %d bytes\n", buffer_size);

    int res = emit_to_rw_run_from_rx(str);

    result = PyLong_FromLong(res);

    return result;
}

static PyMethodDef JitterMethods[] = {
	{"jit", jitter_jit, METH_VARARGS, "Jit a method at runtime"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef jittermodule = {
	PyModuleDef_HEAD_INIT,
	"jitter",
	NULL,
	-1,
	JitterMethods
};

PyMODINIT_FUNC PyInit_jitter(void) {
	PyObject *m;
	m = PyModule_Create(&jittermodule);
	if (m == NULL)
		return NULL;

	JitterError = PyErr_NewException("jitter.error", NULL, NULL);
	Py_INCREF(JitterError);
	PyModule_AddObject(m, "error", JitterError);
	return m;
}

int main(int argc, char *argv[]) {
	PyImport_AppendInittab("jitter", PyInit_jitter);

	Py_SetProgramName(argv[0]);

	Py_Initialize();

	PyImport_ImportModule("jitter");
}
~~~

The first part of the code is very similar to the blog linked above. However, the main part is the interface to python.
The JitterMethods defines a jit method which will be exposed. The jitter_jit exposes the entry point to which we actually send the byte code to be jitted. This method takes in a series of bytes and generates the code for it.

To help with the build I defined a file called build.sh and put the following code in it.

~~~bash
python3 setup.py build
~~~

Running this should generate an .so file on linux. You should be able to see it in the build folder.
Copy the .so file to the root folder so its easier to run it or you can run it from the Interactive prompt too.

So here is an example of how to run it without any other api support.

~~~python
l = [hex(0x48), hex(0x89), hex(0xf8), hex(0x48), hex(0x83), hex(0xc0), hex(0x04), hex(0xc3)]
import binascii

b = bytes(int(x, 16) for x in l)
import jitter
val = jitter.jit(b)
print(val)
~~~

The code is the assembly encoding of moving 50 to eax. That's the return value from the function.

~~~assembly
mov eax, 50
ret
~~~

Pretty slick huh !!!

# Pyasm2
Now we change gears and use an api which will generate the bytecode assembly for us. I used a library called PyAsm2 which is a very small library. However, it wasn't working for me in raw form so I decided to change it a little bit.
[PyAsm2 Github](https://github.com/ssarangi/pyasm2). The code is originally from jbremer but I made some modifications to it for it to be able to give us back raw bytes. jbremer's code used to return strings which was a little painful to work with.

~~~python
from x86 import *
import jitter
import binascii

insts = []
insts.append(mov(eax, 50))
insts.append(ret())

x86bytes = []
for inst in insts:
    x86bytes += inst.bytes()

x86_bytes = bytes(int(x, 16) for x in x86bytes)
val = jitter.jit(x86_bytes)
print("Final return value: %s" % val)
~~~

So that's it. My main goal is to be able to generate code using PyAsm2 and feed it to this simple jitter. However, I am still far from it. Right now, I am tweaking pyasm2 to understand its architecture and it will take some time before I manage to do it but my project [SpiderJIT](https://github.com/ssarangi/spiderjit) is where I am trying to do this.

I will write some more blog posts on how that is going once I make some significant progress.
