# Skulpt



## Problem Statement
- Pull request to any public github repo
- Cannot be a directly managed project
- Has to be interesting



## Challenges
- Time Constraints
- Project merging constraints
- Work to be done at arm's length



## Time Spent
- 2 days for understanding code base
- 2 days for implementing the debugger
- 1 day for adhering to coding standards



## Similar Projects & Why Skulpt
- Brython
- Py.js
- pypy.js
- Rusthon



## High Level Architecture
- Lexer / Parser (Mirrors the current code from CPython)
- Code Generator
- Builtin Handling



## Code Generation - Input
```
def foo(a, b):
    a = a + 1
    print("foo")

def goo(c, d):
    foo(c+1, d+1)
    print("goo")

def boo(e, f):
    goo(e+1, f+1)
    print("boo")

boo(1, 2)
```



## Code Generation - Output (No Debug)
```javascript
/*     1 */ Sk.execStart = Sk.lastYield = new Date();
/*     2 */ var $scope0 = (function($modname) {
/*     3 */     var $wakeFromSuspension = function() {
/*     4 */         var susp = $scope0.wakingSuspension;
/*     5 */         delete $scope0.wakingSuspension;
/*     6 */         $blk = susp.$blk;
/*     7 */         $loc = susp.$loc;
/*     8 */         $gbl = susp.$gbl;
/*     9 */         $exc = susp.$exc;
/*    10 */         $err = susp.$err;
/*    11 */         currLineNo = susp.lineno;
/*    12 */         currColNo = susp.colno;
/*    13 */         Sk.lastYield = Date.now();
/*    14 */         $loadname31 = susp.$tmps.$loadname31;
/*    15 */         try {
/*    16 */             $ret = susp.child.resume();
/*    17 */         } catch (err) {
/*    18 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*    19 */                 err = new Sk.builtin.ExternalError(err);
/*    20 */             }
/*    21 */             err.traceback.push({
/*    22 */                 lineno: currLineNo,
/*    23 */                 colno: currColNo,
/*    24 */                 filename: '<stdin>.py'
/*    25 */             });
/*    26 */             if ($exc.length > 0) {
/*    27 */                 $err = err;
/*    28 */                 $blk = $exc.pop();
/*    29 */             } else {
/*    30 */                 throw err;
/*    31 */             }
/*    32 */         }
/*    33 */     };
/*    34 */     var $saveSuspension = function(child, filename, lineno, colno) {
/*    35 */         var susp = new Sk.misceval.Suspension();
/*    36 */         susp.child = child;
/*    37 */         susp.resume = function() {
/*    38 */             $scope0.wakingSuspension = susp;
/*    39 */             return $scope0();
/*    40 */         };
/*    41 */         susp.data = susp.child.data;
/*    42 */         susp.$blk = $blk;
/*    43 */         susp.$loc = $loc;
/*    44 */         susp.$gbl = $gbl;
/*    45 */         susp.$exc = $exc;
/*    46 */         susp.$err = $err;
/*    47 */         susp.filename = filename;
/*    48 */         susp.lineno = lineno;
/*    49 */         susp.colno = colno;
/*    50 */         susp.optional = child.optional;
/*    51 */         susp.$tmps = {
/*    52 */             "$loadname31": $loadname31
/*    53 */         };
/*    54 */         return susp;
/*    55 */     };
/*    56 */     var $gbl = {},
/*    57 */         $blk = 0,
/*    58 */         $exc = [],
/*    59 */         $loc = $gbl,
/*    60 */         $err = undefined;
/*    61 */     $gbl.__name__ = $modname;
/*    62 */     $loc.__file__ = new Sk.builtins.str('<stdin>.py');
/*    63 */     var $ret = undefined,
/*    64 */         currLineNo = undefined,
/*    65 */         currColNo = undefined;
/*    66 */     if ($scope0.wakingSuspension !== undefined) {
/*    67 */         $wakeFromSuspension();
/*    68 */     }
/*    69 */     if (Sk.retainGlobals) {
/*    70 */         if (Sk.globals) {
/*    71 */             $gbl = Sk.globals;
/*    72 */             Sk.globals = $gbl;
/*    73 */             $loc = $gbl;
/*    74 */         } else {
/*    75 */             Sk.globals = $gbl;
/*    76 */         }
/*    77 */     } else {
/*    78 */         Sk.globals = $gbl;
/*    79 */     }
/*    80 */     while (true) {
/*    81 */         try {
/*    82 */             switch ($blk) {
/*    83 */             case 0:
/*    84 */                 /* --- module entry --- */
/*    85 */                 //
/*    86 */                 // line 1:
/*    87 */                 // def foo(a, b):
/*    88 */                 // ^
/*    89 */                 //
/*    90 */                 currLineNo = 1;
/*    91 */                 currColNo = 0;
/*    92 */ 
/*    93 */                 $scope1.co_name = new Sk.builtins['str']('foo');
/*    94 */                 $scope1.co_varnames = ['a', 'b'];
/*    95 */                 var $funcobj8 = new Sk.builtins['function']($scope1, $gbl);
/*    96 */                 $loc.foo = $funcobj8;
/*    97 */                 //
/*    98 */                 // line 6:
/*    99 */                 // def goo(c, d):
/*   100 */                 // ^
/*   101 */                 //
/*   102 */                 currLineNo = 6;
/*   103 */                 currColNo = 0;
/*   104 */ 
/*   105 */                 $scope9.co_name = new Sk.builtins['str']('goo');
/*   106 */                 $scope9.co_varnames = ['c', 'd'];
/*   107 */                 var $funcobj19 = new Sk.builtins['function']($scope9, $gbl);
/*   108 */                 $loc.goo = $funcobj19;
/*   109 */                 //
/*   110 */                 // line 11:
/*   111 */                 // def boo(e, f):
/*   112 */                 // ^
/*   113 */                 //
/*   114 */                 currLineNo = 11;
/*   115 */                 currColNo = 0;
/*   116 */ 
/*   117 */                 $scope20.co_name = new Sk.builtins['str']('boo');
/*   118 */                 $scope20.co_varnames = ['e', 'f'];
/*   119 */                 var $funcobj30 = new Sk.builtins['function']($scope20, $gbl);
/*   120 */                 $loc.boo = $funcobj30;
/*   121 */                 //
/*   122 */                 // line 16:
/*   123 */                 // boo(1, 2)
/*   124 */                 // ^
/*   125 */                 //
/*   126 */                 currLineNo = 16;
/*   127 */                 currColNo = 0;
/*   128 */ 
/*   129 */                 var $loadname31 = $loc.boo !== undefined ? $loc.boo : Sk.misceval.loadname('boo', $gbl);;
/*   130 */                 $ret;
/*   131 */                 $ret = Sk.misceval.callsimOrSuspend($loadname31, new Sk.builtin.int_(1), new Sk.builtin.int_(2));
/*   132 */                 $blk = 1; /* allowing case fallthrough */
/*   133 */             case 1:
/*   134 */                 /* --- function return or resume suspension --- */
/*   135 */                 if ($ret && $ret.isSuspension) {
/*   136 */                     return $saveSuspension($ret, '<stdin>.py', 16, 0);
/*   137 */                 }
/*   138 */                 var $call32 = $ret;
/*   139 */                 //
/*   140 */                 // line 16:
/*   141 */                 // boo(1, 2)
/*   142 */                 // ^
/*   143 */                 //
/*   144 */                 currLineNo = 16;
/*   145 */                 currColNo = 0;
/*   146 */ 
/*   147 */                 return $loc;
/*   148 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   149 */             }
/*   150 */         } catch (err) {
/*   151 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   152 */                 err = new Sk.builtin.ExternalError(err);
/*   153 */             }
/*   154 */             err.traceback.push({
/*   155 */                 lineno: currLineNo,
/*   156 */                 colno: currColNo,
/*   157 */                 filename: '<stdin>.py'
/*   158 */             });
/*   159 */             if ($exc.length > 0) {
/*   160 */                 $err = err;
/*   161 */                 $blk = $exc.pop();
/*   162 */                 continue;
/*   163 */             } else {
/*   164 */                 throw err;
/*   165 */             }
/*   166 */         }
/*   167 */     }
/*   168 */ });
/*   169 */ var $scope1 = (function $foo2$(a, b) {
/*   170 */     var $blk = 0,
/*   171 */         $exc = [],
/*   172 */         $loc = {},
/*   173 */         $gbl = this,
/*   174 */         $err = undefined,
/*   175 */         $ret = undefined,
/*   176 */         currLineNo = undefined,
/*   177 */         currColNo = undefined;
/*   178 */     if ($scope1.wakingSuspension !== undefined) {
/*   179 */         $wakeFromSuspension();
/*   180 */     } else {
/*   181 */         Sk.builtin.pyCheckArgs("foo", arguments, 2, 2, false, false);
/*   182 */     }
/*   183 */     while (true) {
/*   184 */         try {
/*   185 */             switch ($blk) {
/*   186 */             case 0:
/*   187 */                 /* --- codeobj entry --- */
/*   188 */                 if (a === undefined) {
/*   189 */                     throw new Sk.builtin.UnboundLocalError('local variable \'a\' referenced before assignment');
/*   190 */                 }
/*   191 */                 if (b === undefined) {
/*   192 */                     throw new Sk.builtin.UnboundLocalError('local variable \'b\' referenced before assignment');
/*   193 */                 }
/*   194 */ 
/*   195 */                 //
/*   196 */                 // line 2:
/*   197 */                 //     a = a + 1
/*   198 */                 //     ^
/*   199 */                 //
/*   200 */                 currLineNo = 2;
/*   201 */                 currColNo = 4;
/*   202 */ 
/*   203 */                 if (a === undefined) {
/*   204 */                     throw new Sk.builtin.UnboundLocalError('local variable \'a\' referenced before assignment');
/*   205 */                 }
/*   206 */                 var $binop3 = Sk.abstr.numberBinOp(a, new Sk.builtin.int_(1), 'Add');
/*   207 */                 a = $binop3;
/*   208 */                 //
/*   209 */                 // line 3:
/*   210 */                 //     print(a, b)
/*   211 */                 //     ^
/*   212 */                 //
/*   213 */                 currLineNo = 3;
/*   214 */                 currColNo = 4;
/*   215 */ 
/*   216 */                 if (a === undefined) {
/*   217 */                     throw new Sk.builtin.UnboundLocalError('local variable \'a\' referenced before assignment');
/*   218 */                 }
/*   219 */                 var $elem4 = a;
/*   220 */                 if (b === undefined) {
/*   221 */                     throw new Sk.builtin.UnboundLocalError('local variable \'b\' referenced before assignment');
/*   222 */                 }
/*   223 */                 var $elem5 = b;
/*   224 */                 var $loadtuple6 = new Sk.builtins['tuple']([$elem4, $elem5]);
/*   225 */                 Sk.misceval.print_(new Sk.builtins['str']($loadtuple6).v);
/*   226 */                 Sk.misceval.print_("\n");
/*   227 */                 //
/*   228 */                 // line 4:
/*   229 */                 //     print("foo")
/*   230 */                 //     ^
/*   231 */                 //
/*   232 */                 currLineNo = 4;
/*   233 */                 currColNo = 4;
/*   234 */ 
/*   235 */                 var $str7 = new Sk.builtins['str']('foo');
/*   236 */                 Sk.misceval.print_(new Sk.builtins['str']($str7).v);
/*   237 */                 Sk.misceval.print_("\n");
/*   238 */                 return Sk.builtin.none.none$;
/*   239 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   240 */             }
/*   241 */         } catch (err) {
/*   242 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   243 */                 err = new Sk.builtin.ExternalError(err);
/*   244 */             }
/*   245 */             err.traceback.push({
/*   246 */                 lineno: currLineNo,
/*   247 */                 colno: currColNo,
/*   248 */                 filename: '<stdin>.py'
/*   249 */             });
/*   250 */             if ($exc.length > 0) {
/*   251 */                 $err = err;
/*   252 */                 $blk = $exc.pop();
/*   253 */                 continue;
/*   254 */             } else {
/*   255 */                 throw err;
/*   256 */             }
/*   257 */         }
/*   258 */     }
/*   259 */ });
/*   260 */ var $scope9 = (function $goo10$(c, d) {
/*   261 */     var $wakeFromSuspension = function() {
/*   262 */         var susp = $scope9.wakingSuspension;
/*   263 */         delete $scope9.wakingSuspension;
/*   264 */         $blk = susp.$blk;
/*   265 */         $loc = susp.$loc;
/*   266 */         $gbl = susp.$gbl;
/*   267 */         $exc = susp.$exc;
/*   268 */         $err = susp.$err;
/*   269 */         currLineNo = susp.lineno;
/*   270 */         currColNo = susp.colno;
/*   271 */         Sk.lastYield = Date.now();
/*   272 */         c = susp.$tmps.c;
/*   273 */         d = susp.$tmps.d;
/*   274 */         $loadgbl11 = susp.$tmps.$loadgbl11;
/*   275 */         $binop12 = susp.$tmps.$binop12;
/*   276 */         $binop13 = susp.$tmps.$binop13;
/*   277 */         try {
/*   278 */             $ret = susp.child.resume();
/*   279 */         } catch (err) {
/*   280 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   281 */                 err = new Sk.builtin.ExternalError(err);
/*   282 */             }
/*   283 */             err.traceback.push({
/*   284 */                 lineno: currLineNo,
/*   285 */                 colno: currColNo,
/*   286 */                 filename: '<stdin>.py'
/*   287 */             });
/*   288 */             if ($exc.length > 0) {
/*   289 */                 $err = err;
/*   290 */                 $blk = $exc.pop();
/*   291 */             } else {
/*   292 */                 throw err;
/*   293 */             }
/*   294 */         }
/*   295 */     };
/*   296 */     var $saveSuspension = function(child, filename, lineno, colno) {
/*   297 */         var susp = new Sk.misceval.Suspension();
/*   298 */         susp.child = child;
/*   299 */         susp.resume = function() {
/*   300 */             $scope9.wakingSuspension = susp;
/*   301 */             return $scope9();
/*   302 */         };
/*   303 */         susp.data = susp.child.data;
/*   304 */         susp.$blk = $blk;
/*   305 */         susp.$loc = $loc;
/*   306 */         susp.$gbl = $gbl;
/*   307 */         susp.$exc = $exc;
/*   308 */         susp.$err = $err;
/*   309 */         susp.filename = filename;
/*   310 */         susp.lineno = lineno;
/*   311 */         susp.colno = colno;
/*   312 */         susp.optional = child.optional;
/*   313 */         susp.$tmps = {
/*   314 */             "c": c,
/*   315 */             "d": d,
/*   316 */             "$loadgbl11": $loadgbl11,
/*   317 */             "$binop12": $binop12,
/*   318 */             "$binop13": $binop13
/*   319 */         };
/*   320 */         return susp;
/*   321 */     };
/*   322 */     var $blk = 0,
/*   323 */         $exc = [],
/*   324 */         $loc = {},
/*   325 */         $gbl = this,
/*   326 */         $err = undefined,
/*   327 */         $ret = undefined,
/*   328 */         currLineNo = undefined,
/*   329 */         currColNo = undefined;
/*   330 */     if ($scope9.wakingSuspension !== undefined) {
/*   331 */         $wakeFromSuspension();
/*   332 */     } else {
/*   333 */         Sk.builtin.pyCheckArgs("goo", arguments, 2, 2, false, false);
/*   334 */     }
/*   335 */     while (true) {
/*   336 */         try {
/*   337 */             switch ($blk) {
/*   338 */             case 0:
/*   339 */                 /* --- codeobj entry --- */
/*   340 */                 if (c === undefined) {
/*   341 */                     throw new Sk.builtin.UnboundLocalError('local variable \'c\' referenced before assignment');
/*   342 */                 }
/*   343 */                 if (d === undefined) {
/*   344 */                     throw new Sk.builtin.UnboundLocalError('local variable \'d\' referenced before assignment');
/*   345 */                 }
/*   346 */ 
/*   347 */                 //
/*   348 */                 // line 7:
/*   349 */                 //     foo(c+1, d+1)
/*   350 */                 //     ^
/*   351 */                 //
/*   352 */                 currLineNo = 7;
/*   353 */                 currColNo = 4;
/*   354 */ 
/*   355 */                 var $loadgbl11 = Sk.misceval.loadname('foo', $gbl);
/*   356 */                 if (c === undefined) {
/*   357 */                     throw new Sk.builtin.UnboundLocalError('local variable \'c\' referenced before assignment');
/*   358 */                 }
/*   359 */                 var $binop12 = Sk.abstr.numberBinOp(c, new Sk.builtin.int_(1), 'Add');
/*   360 */                 if (d === undefined) {
/*   361 */                     throw new Sk.builtin.UnboundLocalError('local variable \'d\' referenced before assignment');
/*   362 */                 }
/*   363 */                 var $binop13 = Sk.abstr.numberBinOp(d, new Sk.builtin.int_(1), 'Add');
/*   364 */                 $ret;
/*   365 */                 $ret = Sk.misceval.callsimOrSuspend($loadgbl11, $binop12, $binop13);
/*   366 */                 $blk = 1; /* allowing case fallthrough */
/*   367 */             case 1:
/*   368 */                 /* --- function return or resume suspension --- */
/*   369 */                 if ($ret && $ret.isSuspension) {
/*   370 */                     return $saveSuspension($ret, '<stdin>.py', 7, 4);
/*   371 */                 }
/*   372 */                 var $call14 = $ret;
/*   373 */                 //
/*   374 */                 // line 7:
/*   375 */                 //     foo(c+1, d+1)
/*   376 */                 //     ^
/*   377 */                 //
/*   378 */                 currLineNo = 7;
/*   379 */                 currColNo = 4;
/*   380 */ 
/*   381 */ 
/*   382 */                 //
/*   383 */                 // line 8:
/*   384 */                 //     print(c, d)
/*   385 */                 //     ^
/*   386 */                 //
/*   387 */                 currLineNo = 8;
/*   388 */                 currColNo = 4;
/*   389 */ 
/*   390 */                 if (c === undefined) {
/*   391 */                     throw new Sk.builtin.UnboundLocalError('local variable \'c\' referenced before assignment');
/*   392 */                 }
/*   393 */                 var $elem15 = c;
/*   394 */                 if (d === undefined) {
/*   395 */                     throw new Sk.builtin.UnboundLocalError('local variable \'d\' referenced before assignment');
/*   396 */                 }
/*   397 */                 var $elem16 = d;
/*   398 */                 var $loadtuple17 = new Sk.builtins['tuple']([$elem15, $elem16]);
/*   399 */                 Sk.misceval.print_(new Sk.builtins['str']($loadtuple17).v);
/*   400 */                 Sk.misceval.print_("\n");
/*   401 */                 //
/*   402 */                 // line 9:
/*   403 */                 //     print("goo")
/*   404 */                 //     ^
/*   405 */                 //
/*   406 */                 currLineNo = 9;
/*   407 */                 currColNo = 4;
/*   408 */ 
/*   409 */                 var $str18 = new Sk.builtins['str']('goo');
/*   410 */                 Sk.misceval.print_(new Sk.builtins['str']($str18).v);
/*   411 */                 Sk.misceval.print_("\n");
/*   412 */                 return Sk.builtin.none.none$;
/*   413 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   414 */             }
/*   415 */         } catch (err) {
/*   416 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   417 */                 err = new Sk.builtin.ExternalError(err);
/*   418 */             }
/*   419 */             err.traceback.push({
/*   420 */                 lineno: currLineNo,
/*   421 */                 colno: currColNo,
/*   422 */                 filename: '<stdin>.py'
/*   423 */             });
/*   424 */             if ($exc.length > 0) {
/*   425 */                 $err = err;
/*   426 */                 $blk = $exc.pop();
/*   427 */                 continue;
/*   428 */             } else {
/*   429 */                 throw err;
/*   430 */             }
/*   431 */         }
/*   432 */     }
/*   433 */ });
/*   434 */ var $scope20 = (function $boo21$(e, f) {
/*   435 */     var $wakeFromSuspension = function() {
/*   436 */         var susp = $scope20.wakingSuspension;
/*   437 */         delete $scope20.wakingSuspension;
/*   438 */         $blk = susp.$blk;
/*   439 */         $loc = susp.$loc;
/*   440 */         $gbl = susp.$gbl;
/*   441 */         $exc = susp.$exc;
/*   442 */         $err = susp.$err;
/*   443 */         currLineNo = susp.lineno;
/*   444 */         currColNo = susp.colno;
/*   445 */         Sk.lastYield = Date.now();
/*   446 */         e = susp.$tmps.e;
/*   447 */         f = susp.$tmps.f;
/*   448 */         $loadgbl22 = susp.$tmps.$loadgbl22;
/*   449 */         $binop23 = susp.$tmps.$binop23;
/*   450 */         $binop24 = susp.$tmps.$binop24;
/*   451 */         try {
/*   452 */             $ret = susp.child.resume();
/*   453 */         } catch (err) {
/*   454 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   455 */                 err = new Sk.builtin.ExternalError(err);
/*   456 */             }
/*   457 */             err.traceback.push({
/*   458 */                 lineno: currLineNo,
/*   459 */                 colno: currColNo,
/*   460 */                 filename: '<stdin>.py'
/*   461 */             });
/*   462 */             if ($exc.length > 0) {
/*   463 */                 $err = err;
/*   464 */                 $blk = $exc.pop();
/*   465 */             } else {
/*   466 */                 throw err;
/*   467 */             }
/*   468 */         }
/*   469 */     };
/*   470 */     var $saveSuspension = function(child, filename, lineno, colno) {
/*   471 */         var susp = new Sk.misceval.Suspension();
/*   472 */         susp.child = child;
/*   473 */         susp.resume = function() {
/*   474 */             $scope20.wakingSuspension = susp;
/*   475 */             return $scope20();
/*   476 */         };
/*   477 */         susp.data = susp.child.data;
/*   478 */         susp.$blk = $blk;
/*   479 */         susp.$loc = $loc;
/*   480 */         susp.$gbl = $gbl;
/*   481 */         susp.$exc = $exc;
/*   482 */         susp.$err = $err;
/*   483 */         susp.filename = filename;
/*   484 */         susp.lineno = lineno;
/*   485 */         susp.colno = colno;
/*   486 */         susp.optional = child.optional;
/*   487 */         susp.$tmps = {
/*   488 */             "e": e,
/*   489 */             "f": f,
/*   490 */             "$loadgbl22": $loadgbl22,
/*   491 */             "$binop23": $binop23,
/*   492 */             "$binop24": $binop24
/*   493 */         };
/*   494 */         return susp;
/*   495 */     };
/*   496 */     var $blk = 0,
/*   497 */         $exc = [],
/*   498 */         $loc = {},
/*   499 */         $gbl = this,
/*   500 */         $err = undefined,
/*   501 */         $ret = undefined,
/*   502 */         currLineNo = undefined,
/*   503 */         currColNo = undefined;
/*   504 */     if ($scope20.wakingSuspension !== undefined) {
/*   505 */         $wakeFromSuspension();
/*   506 */     } else {
/*   507 */         Sk.builtin.pyCheckArgs("boo", arguments, 2, 2, false, false);
/*   508 */     }
/*   509 */     while (true) {
/*   510 */         try {
/*   511 */             switch ($blk) {
/*   512 */             case 0:
/*   513 */                 /* --- codeobj entry --- */
/*   514 */                 if (e === undefined) {
/*   515 */                     throw new Sk.builtin.UnboundLocalError('local variable \'e\' referenced before assignment');
/*   516 */                 }
/*   517 */                 if (f === undefined) {
/*   518 */                     throw new Sk.builtin.UnboundLocalError('local variable \'f\' referenced before assignment');
/*   519 */                 }
/*   520 */ 
/*   521 */                 //
/*   522 */                 // line 12:
/*   523 */                 //     goo(e+1, f+1)
/*   524 */                 //     ^
/*   525 */                 //
/*   526 */                 currLineNo = 12;
/*   527 */                 currColNo = 4;
/*   528 */ 
/*   529 */                 var $loadgbl22 = Sk.misceval.loadname('goo', $gbl);
/*   530 */                 if (e === undefined) {
/*   531 */                     throw new Sk.builtin.UnboundLocalError('local variable \'e\' referenced before assignment');
/*   532 */                 }
/*   533 */                 var $binop23 = Sk.abstr.numberBinOp(e, new Sk.builtin.int_(1), 'Add');
/*   534 */                 if (f === undefined) {
/*   535 */                     throw new Sk.builtin.UnboundLocalError('local variable \'f\' referenced before assignment');
/*   536 */                 }
/*   537 */                 var $binop24 = Sk.abstr.numberBinOp(f, new Sk.builtin.int_(1), 'Add');
/*   538 */                 $ret;
/*   539 */                 $ret = Sk.misceval.callsimOrSuspend($loadgbl22, $binop23, $binop24);
/*   540 */                 $blk = 1; /* allowing case fallthrough */
/*   541 */             case 1:
/*   542 */                 /* --- function return or resume suspension --- */
/*   543 */                 if ($ret && $ret.isSuspension) {
/*   544 */                     return $saveSuspension($ret, '<stdin>.py', 12, 4);
/*   545 */                 }
/*   546 */                 var $call25 = $ret;
/*   547 */                 //
/*   548 */                 // line 12:
/*   549 */                 //     goo(e+1, f+1)
/*   550 */                 //     ^
/*   551 */                 //
/*   552 */                 currLineNo = 12;
/*   553 */                 currColNo = 4;
/*   554 */ 
/*   555 */ 
/*   556 */                 //
/*   557 */                 // line 13:
/*   558 */                 //     print(e, f)
/*   559 */                 //     ^
/*   560 */                 //
/*   561 */                 currLineNo = 13;
/*   562 */                 currColNo = 4;
/*   563 */ 
/*   564 */                 if (e === undefined) {
/*   565 */                     throw new Sk.builtin.UnboundLocalError('local variable \'e\' referenced before assignment');
/*   566 */                 }
/*   567 */                 var $elem26 = e;
/*   568 */                 if (f === undefined) {
/*   569 */                     throw new Sk.builtin.UnboundLocalError('local variable \'f\' referenced before assignment');
/*   570 */                 }
/*   571 */                 var $elem27 = f;
/*   572 */                 var $loadtuple28 = new Sk.builtins['tuple']([$elem26, $elem27]);
/*   573 */                 Sk.misceval.print_(new Sk.builtins['str']($loadtuple28).v);
/*   574 */                 Sk.misceval.print_("\n");
/*   575 */                 //
/*   576 */                 // line 14:
/*   577 */                 //     print("boo")
/*   578 */                 //     ^
/*   579 */                 //
/*   580 */                 currLineNo = 14;
/*   581 */                 currColNo = 4;
/*   582 */ 
/*   583 */                 var $str29 = new Sk.builtins['str']('boo');
/*   584 */                 Sk.misceval.print_(new Sk.builtins['str']($str29).v);
/*   585 */                 Sk.misceval.print_("\n");
/*   586 */                 return Sk.builtin.none.none$;
/*   587 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   588 */             }
/*   589 */         } catch (err) {
/*   590 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   591 */                 err = new Sk.builtin.ExternalError(err);
/*   592 */             }
/*   593 */             err.traceback.push({
/*   594 */                 lineno: currLineNo,
/*   595 */                 colno: currColNo,
/*   596 */                 filename: '<stdin>.py'
/*   597 */             });
/*   598 */             if ($exc.length > 0) {
/*   599 */                 $err = err;
/*   600 */                 $blk = $exc.pop();
/*   601 */                 continue;
/*   602 */             } else {
/*   603 */                 throw err;
/*   604 */             }
/*   605 */         }
/*   606 */     }
/*   607 */ });
```



## Code Generation - Output (With Debug)
```javascript
/*     1 */ Sk.execStart = Sk.lastYield = new Date();
/*     2 */ var $scope0 = (function($modname) {
/*     3 */     var $wakeFromSuspension = function() {
/*     4 */         var susp = $scope0.wakingSuspension;
/*     5 */         delete $scope0.wakingSuspension;
/*     6 */         $blk = susp.$blk;
/*     7 */         $loc = susp.$loc;
/*     8 */         $gbl = susp.$gbl;
/*     9 */         $exc = susp.$exc;
/*    10 */         $err = susp.$err;
/*    11 */         currLineNo = susp.lineno;
/*    12 */         currColNo = susp.colno;
/*    13 */         Sk.lastYield = Date.now();
/*    14 */         $loadname31 = susp.$tmps.$loadname31;
/*    15 */         try {
/*    16 */             $ret = susp.child.resume();
/*    17 */         } catch (err) {
/*    18 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*    19 */                 err = new Sk.builtin.ExternalError(err);
/*    20 */             }
/*    21 */             err.traceback.push({
/*    22 */                 lineno: currLineNo,
/*    23 */                 colno: currColNo,
/*    24 */                 filename: '<stdin>.py'
/*    25 */             });
/*    26 */             if ($exc.length > 0) {
/*    27 */                 $err = err;
/*    28 */                 $blk = $exc.pop();
/*    29 */             } else {
/*    30 */                 throw err;
/*    31 */             }
/*    32 */         }
/*    33 */     };
/*    34 */     var $saveSuspension = function(child, filename, lineno, colno) {
/*    35 */         var susp = new Sk.misceval.Suspension();
/*    36 */         susp.child = child;
/*    37 */         susp.resume = function() {
/*    38 */             $scope0.wakingSuspension = susp;
/*    39 */             return $scope0();
/*    40 */         };
/*    41 */         susp.data = susp.child.data;
/*    42 */         susp.$blk = $blk;
/*    43 */         susp.$loc = $loc;
/*    44 */         susp.$gbl = $gbl;
/*    45 */         susp.$exc = $exc;
/*    46 */         susp.$err = $err;
/*    47 */         susp.filename = filename;
/*    48 */         susp.lineno = lineno;
/*    49 */         susp.colno = colno;
/*    50 */         susp.optional = child.optional;
/*    51 */         susp.$tmps = {
/*    52 */             "$loadname31": $loadname31
/*    53 */         };
/*    54 */         return susp;
/*    55 */     };
/*    56 */     var $gbl = {},
/*    57 */         $blk = 0,
/*    58 */         $exc = [],
/*    59 */         $loc = $gbl,
/*    60 */         $err = undefined;
/*    61 */     $gbl.__name__ = $modname;
/*    62 */     $loc.__file__ = new Sk.builtins.str('<stdin>.py');
/*    63 */     var $ret = undefined,
/*    64 */         currLineNo = undefined,
/*    65 */         currColNo = undefined;
/*    66 */     if ($scope0.wakingSuspension !== undefined) {
/*    67 */         $wakeFromSuspension();
/*    68 */     }
/*    69 */     if (Sk.retainGlobals) {
/*    70 */         if (Sk.globals) {
/*    71 */             $gbl = Sk.globals;
/*    72 */             Sk.globals = $gbl;
/*    73 */             $loc = $gbl;
/*    74 */         } else {
/*    75 */             Sk.globals = $gbl;
/*    76 */         }
/*    77 */     } else {
/*    78 */         Sk.globals = $gbl;
/*    79 */     }
/*    80 */     while (true) {
/*    81 */         try {
/*    82 */             switch ($blk) {
/*    83 */             case 0:
/*    84 */                 /* --- module entry --- */
/*    85 */                 if (Sk.breakpoints('<stdin>.py', 1, 0)) {
/*    86 */                     var $susp = $saveSuspension({
/*    87 */                         data: {
/*    88 */                             type: 'Sk.debug'
/*    89 */                         },
/*    90 */                         resume: function() {}
/*    91 */                     }, '<stdin>.py', 1, 0);
/*    92 */                     $susp.$blk = 1;
/*    93 */                     $susp.optional = true;
/*    94 */                     return $susp;
/*    95 */                 }
/*    96 */                 $blk = 1; /* allowing case fallthrough */
/*    97 */             case 1:
/*    98 */                 /* --- debug breakpoint for line 1 --- */
/*    99 */                 //
/*   100 */                 // line 1:
/*   101 */                 // def foo(a, b):
/*   102 */                 // ^
/*   103 */                 //
/*   104 */                 currLineNo = 1;
/*   105 */                 currColNo = 0;
/*   106 */ 
/*   107 */                 $scope1.co_name = new Sk.builtins['str']('foo');
/*   108 */                 $scope1.co_varnames = ['a', 'b'];
/*   109 */                 var $funcobj8 = new Sk.builtins['function']($scope1, $gbl);
/*   110 */                 $loc.foo = $funcobj8;
/*   111 */                 if (Sk.breakpoints('<stdin>.py', 6, 0)) {
/*   112 */                     var $susp = $saveSuspension({
/*   113 */                         data: {
/*   114 */                             type: 'Sk.debug'
/*   115 */                         },
/*   116 */                         resume: function() {}
/*   117 */                     }, '<stdin>.py', 6, 0);
/*   118 */                     $susp.$blk = 2;
/*   119 */                     $susp.optional = true;
/*   120 */                     return $susp;
/*   121 */                 }
/*   122 */                 $blk = 2; /* allowing case fallthrough */
/*   123 */             case 2:
/*   124 */                 /* --- debug breakpoint for line 6 --- */
/*   125 */                 //
/*   126 */                 // line 6:
/*   127 */                 // def goo(c, d):
/*   128 */                 // ^
/*   129 */                 //
/*   130 */                 currLineNo = 6;
/*   131 */                 currColNo = 0;
/*   132 */ 
/*   133 */                 $scope9.co_name = new Sk.builtins['str']('goo');
/*   134 */                 $scope9.co_varnames = ['c', 'd'];
/*   135 */                 var $funcobj19 = new Sk.builtins['function']($scope9, $gbl);
/*   136 */                 $loc.goo = $funcobj19;
/*   137 */                 if (Sk.breakpoints('<stdin>.py', 11, 0)) {
/*   138 */                     var $susp = $saveSuspension({
/*   139 */                         data: {
/*   140 */                             type: 'Sk.debug'
/*   141 */                         },
/*   142 */                         resume: function() {}
/*   143 */                     }, '<stdin>.py', 11, 0);
/*   144 */                     $susp.$blk = 3;
/*   145 */                     $susp.optional = true;
/*   146 */                     return $susp;
/*   147 */                 }
/*   148 */                 $blk = 3; /* allowing case fallthrough */
/*   149 */             case 3:
/*   150 */                 /* --- debug breakpoint for line 11 --- */
/*   151 */                 //
/*   152 */                 // line 11:
/*   153 */                 // def boo(e, f):
/*   154 */                 // ^
/*   155 */                 //
/*   156 */                 currLineNo = 11;
/*   157 */                 currColNo = 0;
/*   158 */ 
/*   159 */                 $scope20.co_name = new Sk.builtins['str']('boo');
/*   160 */                 $scope20.co_varnames = ['e', 'f'];
/*   161 */                 var $funcobj30 = new Sk.builtins['function']($scope20, $gbl);
/*   162 */                 $loc.boo = $funcobj30;
/*   163 */                 if (Sk.breakpoints('<stdin>.py', 16, 0)) {
/*   164 */                     var $susp = $saveSuspension({
/*   165 */                         data: {
/*   166 */                             type: 'Sk.debug'
/*   167 */                         },
/*   168 */                         resume: function() {}
/*   169 */                     }, '<stdin>.py', 16, 0);
/*   170 */                     $susp.$blk = 4;
/*   171 */                     $susp.optional = true;
/*   172 */                     return $susp;
/*   173 */                 }
/*   174 */                 $blk = 4; /* allowing case fallthrough */
/*   175 */             case 4:
/*   176 */                 /* --- debug breakpoint for line 16 --- */
/*   177 */                 //
/*   178 */                 // line 16:
/*   179 */                 // boo(1, 2)
/*   180 */                 // ^
/*   181 */                 //
/*   182 */                 currLineNo = 16;
/*   183 */                 currColNo = 0;
/*   184 */ 
/*   185 */                 var $loadname31 = $loc.boo !== undefined ? $loc.boo : Sk.misceval.loadname('boo', $gbl);;
/*   186 */                 $ret;
/*   187 */                 $ret = Sk.misceval.callsimOrSuspend($loadname31, new Sk.builtin.int_(1), new Sk.builtin.int_(2));
/*   188 */                 $blk = 5; /* allowing case fallthrough */
/*   189 */             case 5:
/*   190 */                 /* --- function return or resume suspension --- */
/*   191 */                 if ($ret && $ret.isSuspension) {
/*   192 */                     return $saveSuspension($ret, '<stdin>.py', 16, 0);
/*   193 */                 }
/*   194 */                 var $call32 = $ret;
/*   195 */                 //
/*   196 */                 // line 16:
/*   197 */                 // boo(1, 2)
/*   198 */                 // ^
/*   199 */                 //
/*   200 */                 currLineNo = 16;
/*   201 */                 currColNo = 0;
/*   202 */ 
/*   203 */                 return $loc;
/*   204 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   205 */             }
/*   206 */         } catch (err) {
/*   207 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   208 */                 err = new Sk.builtin.ExternalError(err);
/*   209 */             }
/*   210 */             err.traceback.push({
/*   211 */                 lineno: currLineNo,
/*   212 */                 colno: currColNo,
/*   213 */                 filename: '<stdin>.py'
/*   214 */             });
/*   215 */             if ($exc.length > 0) {
/*   216 */                 $err = err;
/*   217 */                 $blk = $exc.pop();
/*   218 */                 continue;
/*   219 */             } else {
/*   220 */                 throw err;
/*   221 */             }
/*   222 */         }
/*   223 */     }
/*   224 */ });
/*   225 */ var $scope1 = (function $foo2$(a, b) {
/*   226 */     var $wakeFromSuspension = function() {
/*   227 */         var susp = $scope1.wakingSuspension;
/*   228 */         delete $scope1.wakingSuspension;
/*   229 */         $blk = susp.$blk;
/*   230 */         $loc = susp.$loc;
/*   231 */         $gbl = susp.$gbl;
/*   232 */         $exc = susp.$exc;
/*   233 */         $err = susp.$err;
/*   234 */         currLineNo = susp.lineno;
/*   235 */         currColNo = susp.colno;
/*   236 */         Sk.lastYield = Date.now();
/*   237 */         a = susp.$tmps.a;
/*   238 */         b = susp.$tmps.b;
/*   239 */         try {
/*   240 */             $ret = susp.child.resume();
/*   241 */         } catch (err) {
/*   242 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   243 */                 err = new Sk.builtin.ExternalError(err);
/*   244 */             }
/*   245 */             err.traceback.push({
/*   246 */                 lineno: currLineNo,
/*   247 */                 colno: currColNo,
/*   248 */                 filename: '<stdin>.py'
/*   249 */             });
/*   250 */             if ($exc.length > 0) {
/*   251 */                 $err = err;
/*   252 */                 $blk = $exc.pop();
/*   253 */             } else {
/*   254 */                 throw err;
/*   255 */             }
/*   256 */         }
/*   257 */     };
/*   258 */     var $saveSuspension = function(child, filename, lineno, colno) {
/*   259 */         var susp = new Sk.misceval.Suspension();
/*   260 */         susp.child = child;
/*   261 */         susp.resume = function() {
/*   262 */             $scope1.wakingSuspension = susp;
/*   263 */             return $scope1();
/*   264 */         };
/*   265 */         susp.data = susp.child.data;
/*   266 */         susp.$blk = $blk;
/*   267 */         susp.$loc = $loc;
/*   268 */         susp.$gbl = $gbl;
/*   269 */         susp.$exc = $exc;
/*   270 */         susp.$err = $err;
/*   271 */         susp.filename = filename;
/*   272 */         susp.lineno = lineno;
/*   273 */         susp.colno = colno;
/*   274 */         susp.optional = child.optional;
/*   275 */         susp.$tmps = {
/*   276 */             "a": a,
/*   277 */             "b": b
/*   278 */         };
/*   279 */         return susp;
/*   280 */     };
/*   281 */     var $blk = 0,
/*   282 */         $exc = [],
/*   283 */         $loc = {},
/*   284 */         $gbl = this,
/*   285 */         $err = undefined,
/*   286 */         $ret = undefined,
/*   287 */         currLineNo = undefined,
/*   288 */         currColNo = undefined;
/*   289 */     if ($scope1.wakingSuspension !== undefined) {
/*   290 */         $wakeFromSuspension();
/*   291 */     } else {
/*   292 */         Sk.builtin.pyCheckArgs("foo", arguments, 2, 2, false, false);
/*   293 */     }
/*   294 */     while (true) {
/*   295 */         try {
/*   296 */             switch ($blk) {
/*   297 */             case 0:
/*   298 */                 /* --- codeobj entry --- */
/*   299 */                 if (a === undefined) {
/*   300 */                     throw new Sk.builtin.UnboundLocalError('local variable \'a\' referenced before assignment');
/*   301 */                 }
/*   302 */                 if (b === undefined) {
/*   303 */                     throw new Sk.builtin.UnboundLocalError('local variable \'b\' referenced before assignment');
/*   304 */                 }
/*   305 */                 if (Sk.breakpoints('<stdin>.py', 2, 4)) {
/*   306 */                     var $susp = $saveSuspension({
/*   307 */                         data: {
/*   308 */                             type: 'Sk.debug'
/*   309 */                         },
/*   310 */                         resume: function() {}
/*   311 */                     }, '<stdin>.py', 2, 4);
/*   312 */                     $susp.$blk = 1;
/*   313 */                     $susp.optional = true;
/*   314 */                     return $susp;
/*   315 */                 }
/*   316 */                 $blk = 1; /* allowing case fallthrough */
/*   317 */             case 1:
/*   318 */                 /* --- debug breakpoint for line 2 --- */
/*   319 */                 //
/*   320 */                 // line 2:
/*   321 */                 //     a = a + 1
/*   322 */                 //     ^
/*   323 */                 //
/*   324 */                 currLineNo = 2;
/*   325 */                 currColNo = 4;
/*   326 */ 
/*   327 */                 if (a === undefined) {
/*   328 */                     throw new Sk.builtin.UnboundLocalError('local variable \'a\' referenced before assignment');
/*   329 */                 }
/*   330 */                 var $binop3 = Sk.abstr.numberBinOp(a, new Sk.builtin.int_(1), 'Add');
/*   331 */                 a = $binop3;
/*   332 */                 if (Sk.breakpoints('<stdin>.py', 3, 4)) {
/*   333 */                     var $susp = $saveSuspension({
/*   334 */                         data: {
/*   335 */                             type: 'Sk.debug'
/*   336 */                         },
/*   337 */                         resume: function() {}
/*   338 */                     }, '<stdin>.py', 3, 4);
/*   339 */                     $susp.$blk = 2;
/*   340 */                     $susp.optional = true;
/*   341 */                     return $susp;
/*   342 */                 }
/*   343 */                 $blk = 2; /* allowing case fallthrough */
/*   344 */             case 2:
/*   345 */                 /* --- debug breakpoint for line 3 --- */
/*   346 */                 //
/*   347 */                 // line 3:
/*   348 */                 //     print(a, b)
/*   349 */                 //     ^
/*   350 */                 //
/*   351 */                 currLineNo = 3;
/*   352 */                 currColNo = 4;
/*   353 */ 
/*   354 */                 if (a === undefined) {
/*   355 */                     throw new Sk.builtin.UnboundLocalError('local variable \'a\' referenced before assignment');
/*   356 */                 }
/*   357 */                 var $elem4 = a;
/*   358 */                 if (b === undefined) {
/*   359 */                     throw new Sk.builtin.UnboundLocalError('local variable \'b\' referenced before assignment');
/*   360 */                 }
/*   361 */                 var $elem5 = b;
/*   362 */                 var $loadtuple6 = new Sk.builtins['tuple']([$elem4, $elem5]);
/*   363 */                 Sk.misceval.print_(new Sk.builtins['str']($loadtuple6).v);
/*   364 */                 Sk.misceval.print_("\n");
/*   365 */                 if (Sk.breakpoints('<stdin>.py', 4, 4)) {
/*   366 */                     var $susp = $saveSuspension({
/*   367 */                         data: {
/*   368 */                             type: 'Sk.debug'
/*   369 */                         },
/*   370 */                         resume: function() {}
/*   371 */                     }, '<stdin>.py', 4, 4);
/*   372 */                     $susp.$blk = 3;
/*   373 */                     $susp.optional = true;
/*   374 */                     return $susp;
/*   375 */                 }
/*   376 */                 $blk = 3; /* allowing case fallthrough */
/*   377 */             case 3:
/*   378 */                 /* --- debug breakpoint for line 4 --- */
/*   379 */                 //
/*   380 */                 // line 4:
/*   381 */                 //     print("foo")
/*   382 */                 //     ^
/*   383 */                 //
/*   384 */                 currLineNo = 4;
/*   385 */                 currColNo = 4;
/*   386 */ 
/*   387 */                 var $str7 = new Sk.builtins['str']('foo');
/*   388 */                 Sk.misceval.print_(new Sk.builtins['str']($str7).v);
/*   389 */                 Sk.misceval.print_("\n");
/*   390 */                 return Sk.builtin.none.none$;
/*   391 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   392 */             }
/*   393 */         } catch (err) {
/*   394 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   395 */                 err = new Sk.builtin.ExternalError(err);
/*   396 */             }
/*   397 */             err.traceback.push({
/*   398 */                 lineno: currLineNo,
/*   399 */                 colno: currColNo,
/*   400 */                 filename: '<stdin>.py'
/*   401 */             });
/*   402 */             if ($exc.length > 0) {
/*   403 */                 $err = err;
/*   404 */                 $blk = $exc.pop();
/*   405 */                 continue;
/*   406 */             } else {
/*   407 */                 throw err;
/*   408 */             }
/*   409 */         }
/*   410 */     }
/*   411 */ });
/*   412 */ var $scope9 = (function $goo10$(c, d) {
/*   413 */     var $wakeFromSuspension = function() {
/*   414 */         var susp = $scope9.wakingSuspension;
/*   415 */         delete $scope9.wakingSuspension;
/*   416 */         $blk = susp.$blk;
/*   417 */         $loc = susp.$loc;
/*   418 */         $gbl = susp.$gbl;
/*   419 */         $exc = susp.$exc;
/*   420 */         $err = susp.$err;
/*   421 */         currLineNo = susp.lineno;
/*   422 */         currColNo = susp.colno;
/*   423 */         Sk.lastYield = Date.now();
/*   424 */         c = susp.$tmps.c;
/*   425 */         d = susp.$tmps.d;
/*   426 */         $loadgbl11 = susp.$tmps.$loadgbl11;
/*   427 */         $binop12 = susp.$tmps.$binop12;
/*   428 */         $binop13 = susp.$tmps.$binop13;
/*   429 */         try {
/*   430 */             $ret = susp.child.resume();
/*   431 */         } catch (err) {
/*   432 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   433 */                 err = new Sk.builtin.ExternalError(err);
/*   434 */             }
/*   435 */             err.traceback.push({
/*   436 */                 lineno: currLineNo,
/*   437 */                 colno: currColNo,
/*   438 */                 filename: '<stdin>.py'
/*   439 */             });
/*   440 */             if ($exc.length > 0) {
/*   441 */                 $err = err;
/*   442 */                 $blk = $exc.pop();
/*   443 */             } else {
/*   444 */                 throw err;
/*   445 */             }
/*   446 */         }
/*   447 */     };
/*   448 */     var $saveSuspension = function(child, filename, lineno, colno) {
/*   449 */         var susp = new Sk.misceval.Suspension();
/*   450 */         susp.child = child;
/*   451 */         susp.resume = function() {
/*   452 */             $scope9.wakingSuspension = susp;
/*   453 */             return $scope9();
/*   454 */         };
/*   455 */         susp.data = susp.child.data;
/*   456 */         susp.$blk = $blk;
/*   457 */         susp.$loc = $loc;
/*   458 */         susp.$gbl = $gbl;
/*   459 */         susp.$exc = $exc;
/*   460 */         susp.$err = $err;
/*   461 */         susp.filename = filename;
/*   462 */         susp.lineno = lineno;
/*   463 */         susp.colno = colno;
/*   464 */         susp.optional = child.optional;
/*   465 */         susp.$tmps = {
/*   466 */             "c": c,
/*   467 */             "d": d,
/*   468 */             "$loadgbl11": $loadgbl11,
/*   469 */             "$binop12": $binop12,
/*   470 */             "$binop13": $binop13
/*   471 */         };
/*   472 */         return susp;
/*   473 */     };
/*   474 */     var $blk = 0,
/*   475 */         $exc = [],
/*   476 */         $loc = {},
/*   477 */         $gbl = this,
/*   478 */         $err = undefined,
/*   479 */         $ret = undefined,
/*   480 */         currLineNo = undefined,
/*   481 */         currColNo = undefined;
/*   482 */     if ($scope9.wakingSuspension !== undefined) {
/*   483 */         $wakeFromSuspension();
/*   484 */     } else {
/*   485 */         Sk.builtin.pyCheckArgs("goo", arguments, 2, 2, false, false);
/*   486 */     }
/*   487 */     while (true) {
/*   488 */         try {
/*   489 */             switch ($blk) {
/*   490 */             case 0:
/*   491 */                 /* --- codeobj entry --- */
/*   492 */                 if (c === undefined) {
/*   493 */                     throw new Sk.builtin.UnboundLocalError('local variable \'c\' referenced before assignment');
/*   494 */                 }
/*   495 */                 if (d === undefined) {
/*   496 */                     throw new Sk.builtin.UnboundLocalError('local variable \'d\' referenced before assignment');
/*   497 */                 }
/*   498 */                 if (Sk.breakpoints('<stdin>.py', 7, 4)) {
/*   499 */                     var $susp = $saveSuspension({
/*   500 */                         data: {
/*   501 */                             type: 'Sk.debug'
/*   502 */                         },
/*   503 */                         resume: function() {}
/*   504 */                     }, '<stdin>.py', 7, 4);
/*   505 */                     $susp.$blk = 1;
/*   506 */                     $susp.optional = true;
/*   507 */                     return $susp;
/*   508 */                 }
/*   509 */                 $blk = 1; /* allowing case fallthrough */
/*   510 */             case 1:
/*   511 */                 /* --- debug breakpoint for line 7 --- */
/*   512 */                 //
/*   513 */                 // line 7:
/*   514 */                 //     foo(c+1, d+1)
/*   515 */                 //     ^
/*   516 */                 //
/*   517 */                 currLineNo = 7;
/*   518 */                 currColNo = 4;
/*   519 */ 
/*   520 */                 var $loadgbl11 = Sk.misceval.loadname('foo', $gbl);
/*   521 */                 if (c === undefined) {
/*   522 */                     throw new Sk.builtin.UnboundLocalError('local variable \'c\' referenced before assignment');
/*   523 */                 }
/*   524 */                 var $binop12 = Sk.abstr.numberBinOp(c, new Sk.builtin.int_(1), 'Add');
/*   525 */                 if (d === undefined) {
/*   526 */                     throw new Sk.builtin.UnboundLocalError('local variable \'d\' referenced before assignment');
/*   527 */                 }
/*   528 */                 var $binop13 = Sk.abstr.numberBinOp(d, new Sk.builtin.int_(1), 'Add');
/*   529 */                 $ret;
/*   530 */                 $ret = Sk.misceval.callsimOrSuspend($loadgbl11, $binop12, $binop13);
/*   531 */                 $blk = 2; /* allowing case fallthrough */
/*   532 */             case 2:
/*   533 */                 /* --- function return or resume suspension --- */
/*   534 */                 if ($ret && $ret.isSuspension) {
/*   535 */                     return $saveSuspension($ret, '<stdin>.py', 7, 4);
/*   536 */                 }
/*   537 */                 var $call14 = $ret;
/*   538 */                 //
/*   539 */                 // line 7:
/*   540 */                 //     foo(c+1, d+1)
/*   541 */                 //     ^
/*   542 */                 //
/*   543 */                 currLineNo = 7;
/*   544 */                 currColNo = 4;
/*   545 */ 
/*   546 */                 if (Sk.breakpoints('<stdin>.py', 8, 4)) {
/*   547 */                     var $susp = $saveSuspension({
/*   548 */                         data: {
/*   549 */                             type: 'Sk.debug'
/*   550 */                         },
/*   551 */                         resume: function() {}
/*   552 */                     }, '<stdin>.py', 8, 4);
/*   553 */                     $susp.$blk = 3;
/*   554 */                     $susp.optional = true;
/*   555 */                     return $susp;
/*   556 */                 }
/*   557 */                 $blk = 3; /* allowing case fallthrough */
/*   558 */             case 3:
/*   559 */                 /* --- debug breakpoint for line 8 --- */
/*   560 */                 //
/*   561 */                 // line 8:
/*   562 */                 //     print(c, d)
/*   563 */                 //     ^
/*   564 */                 //
/*   565 */                 currLineNo = 8;
/*   566 */                 currColNo = 4;
/*   567 */ 
/*   568 */                 if (c === undefined) {
/*   569 */                     throw new Sk.builtin.UnboundLocalError('local variable \'c\' referenced before assignment');
/*   570 */                 }
/*   571 */                 var $elem15 = c;
/*   572 */                 if (d === undefined) {
/*   573 */                     throw new Sk.builtin.UnboundLocalError('local variable \'d\' referenced before assignment');
/*   574 */                 }
/*   575 */                 var $elem16 = d;
/*   576 */                 var $loadtuple17 = new Sk.builtins['tuple']([$elem15, $elem16]);
/*   577 */                 Sk.misceval.print_(new Sk.builtins['str']($loadtuple17).v);
/*   578 */                 Sk.misceval.print_("\n");
/*   579 */                 if (Sk.breakpoints('<stdin>.py', 9, 4)) {
/*   580 */                     var $susp = $saveSuspension({
/*   581 */                         data: {
/*   582 */                             type: 'Sk.debug'
/*   583 */                         },
/*   584 */                         resume: function() {}
/*   585 */                     }, '<stdin>.py', 9, 4);
/*   586 */                     $susp.$blk = 4;
/*   587 */                     $susp.optional = true;
/*   588 */                     return $susp;
/*   589 */                 }
/*   590 */                 $blk = 4; /* allowing case fallthrough */
/*   591 */             case 4:
/*   592 */                 /* --- debug breakpoint for line 9 --- */
/*   593 */                 //
/*   594 */                 // line 9:
/*   595 */                 //     print("goo")
/*   596 */                 //     ^
/*   597 */                 //
/*   598 */                 currLineNo = 9;
/*   599 */                 currColNo = 4;
/*   600 */ 
/*   601 */                 var $str18 = new Sk.builtins['str']('goo');
/*   602 */                 Sk.misceval.print_(new Sk.builtins['str']($str18).v);
/*   603 */                 Sk.misceval.print_("\n");
/*   604 */                 return Sk.builtin.none.none$;
/*   605 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   606 */             }
/*   607 */         } catch (err) {
/*   608 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   609 */                 err = new Sk.builtin.ExternalError(err);
/*   610 */             }
/*   611 */             err.traceback.push({
/*   612 */                 lineno: currLineNo,
/*   613 */                 colno: currColNo,
/*   614 */                 filename: '<stdin>.py'
/*   615 */             });
/*   616 */             if ($exc.length > 0) {
/*   617 */                 $err = err;
/*   618 */                 $blk = $exc.pop();
/*   619 */                 continue;
/*   620 */             } else {
/*   621 */                 throw err;
/*   622 */             }
/*   623 */         }
/*   624 */     }
/*   625 */ });
/*   626 */ var $scope20 = (function $boo21$(e, f) {
/*   627 */     var $wakeFromSuspension = function() {
/*   628 */         var susp = $scope20.wakingSuspension;
/*   629 */         delete $scope20.wakingSuspension;
/*   630 */         $blk = susp.$blk;
/*   631 */         $loc = susp.$loc;
/*   632 */         $gbl = susp.$gbl;
/*   633 */         $exc = susp.$exc;
/*   634 */         $err = susp.$err;
/*   635 */         currLineNo = susp.lineno;
/*   636 */         currColNo = susp.colno;
/*   637 */         Sk.lastYield = Date.now();
/*   638 */         e = susp.$tmps.e;
/*   639 */         f = susp.$tmps.f;
/*   640 */         $loadgbl22 = susp.$tmps.$loadgbl22;
/*   641 */         $binop23 = susp.$tmps.$binop23;
/*   642 */         $binop24 = susp.$tmps.$binop24;
/*   643 */         try {
/*   644 */             $ret = susp.child.resume();
/*   645 */         } catch (err) {
/*   646 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   647 */                 err = new Sk.builtin.ExternalError(err);
/*   648 */             }
/*   649 */             err.traceback.push({
/*   650 */                 lineno: currLineNo,
/*   651 */                 colno: currColNo,
/*   652 */                 filename: '<stdin>.py'
/*   653 */             });
/*   654 */             if ($exc.length > 0) {
/*   655 */                 $err = err;
/*   656 */                 $blk = $exc.pop();
/*   657 */             } else {
/*   658 */                 throw err;
/*   659 */             }
/*   660 */         }
/*   661 */     };
/*   662 */     var $saveSuspension = function(child, filename, lineno, colno) {
/*   663 */         var susp = new Sk.misceval.Suspension();
/*   664 */         susp.child = child;
/*   665 */         susp.resume = function() {
/*   666 */             $scope20.wakingSuspension = susp;
/*   667 */             return $scope20();
/*   668 */         };
/*   669 */         susp.data = susp.child.data;
/*   670 */         susp.$blk = $blk;
/*   671 */         susp.$loc = $loc;
/*   672 */         susp.$gbl = $gbl;
/*   673 */         susp.$exc = $exc;
/*   674 */         susp.$err = $err;
/*   675 */         susp.filename = filename;
/*   676 */         susp.lineno = lineno;
/*   677 */         susp.colno = colno;
/*   678 */         susp.optional = child.optional;
/*   679 */         susp.$tmps = {
/*   680 */             "e": e,
/*   681 */             "f": f,
/*   682 */             "$loadgbl22": $loadgbl22,
/*   683 */             "$binop23": $binop23,
/*   684 */             "$binop24": $binop24
/*   685 */         };
/*   686 */         return susp;
/*   687 */     };
/*   688 */     var $blk = 0,
/*   689 */         $exc = [],
/*   690 */         $loc = {},
/*   691 */         $gbl = this,
/*   692 */         $err = undefined,
/*   693 */         $ret = undefined,
/*   694 */         currLineNo = undefined,
/*   695 */         currColNo = undefined;
/*   696 */     if ($scope20.wakingSuspension !== undefined) {
/*   697 */         $wakeFromSuspension();
/*   698 */     } else {
/*   699 */         Sk.builtin.pyCheckArgs("boo", arguments, 2, 2, false, false);
/*   700 */     }
/*   701 */     while (true) {
/*   702 */         try {
/*   703 */             switch ($blk) {
/*   704 */             case 0:
/*   705 */                 /* --- codeobj entry --- */
/*   706 */                 if (e === undefined) {
/*   707 */                     throw new Sk.builtin.UnboundLocalError('local variable \'e\' referenced before assignment');
/*   708 */                 }
/*   709 */                 if (f === undefined) {
/*   710 */                     throw new Sk.builtin.UnboundLocalError('local variable \'f\' referenced before assignment');
/*   711 */                 }
/*   712 */                 if (Sk.breakpoints('<stdin>.py', 12, 4)) {
/*   713 */                     var $susp = $saveSuspension({
/*   714 */                         data: {
/*   715 */                             type: 'Sk.debug'
/*   716 */                         },
/*   717 */                         resume: function() {}
/*   718 */                     }, '<stdin>.py', 12, 4);
/*   719 */                     $susp.$blk = 1;
/*   720 */                     $susp.optional = true;
/*   721 */                     return $susp;
/*   722 */                 }
/*   723 */                 $blk = 1; /* allowing case fallthrough */
/*   724 */             case 1:
/*   725 */                 /* --- debug breakpoint for line 12 --- */
/*   726 */                 //
/*   727 */                 // line 12:
/*   728 */                 //     goo(e+1, f+1)
/*   729 */                 //     ^
/*   730 */                 //
/*   731 */                 currLineNo = 12;
/*   732 */                 currColNo = 4;
/*   733 */ 
/*   734 */                 var $loadgbl22 = Sk.misceval.loadname('goo', $gbl);
/*   735 */                 if (e === undefined) {
/*   736 */                     throw new Sk.builtin.UnboundLocalError('local variable \'e\' referenced before assignment');
/*   737 */                 }
/*   738 */                 var $binop23 = Sk.abstr.numberBinOp(e, new Sk.builtin.int_(1), 'Add');
/*   739 */                 if (f === undefined) {
/*   740 */                     throw new Sk.builtin.UnboundLocalError('local variable \'f\' referenced before assignment');
/*   741 */                 }
/*   742 */                 var $binop24 = Sk.abstr.numberBinOp(f, new Sk.builtin.int_(1), 'Add');
/*   743 */                 $ret;
/*   744 */                 $ret = Sk.misceval.callsimOrSuspend($loadgbl22, $binop23, $binop24);
/*   745 */                 $blk = 2; /* allowing case fallthrough */
/*   746 */             case 2:
/*   747 */                 /* --- function return or resume suspension --- */
/*   748 */                 if ($ret && $ret.isSuspension) {
/*   749 */                     return $saveSuspension($ret, '<stdin>.py', 12, 4);
/*   750 */                 }
/*   751 */                 var $call25 = $ret;
/*   752 */                 //
/*   753 */                 // line 12:
/*   754 */                 //     goo(e+1, f+1)
/*   755 */                 //     ^
/*   756 */                 //
/*   757 */                 currLineNo = 12;
/*   758 */                 currColNo = 4;
/*   759 */ 
/*   760 */                 if (Sk.breakpoints('<stdin>.py', 13, 4)) {
/*   761 */                     var $susp = $saveSuspension({
/*   762 */                         data: {
/*   763 */                             type: 'Sk.debug'
/*   764 */                         },
/*   765 */                         resume: function() {}
/*   766 */                     }, '<stdin>.py', 13, 4);
/*   767 */                     $susp.$blk = 3;
/*   768 */                     $susp.optional = true;
/*   769 */                     return $susp;
/*   770 */                 }
/*   771 */                 $blk = 3; /* allowing case fallthrough */
/*   772 */             case 3:
/*   773 */                 /* --- debug breakpoint for line 13 --- */
/*   774 */                 //
/*   775 */                 // line 13:
/*   776 */                 //     print(e, f)
/*   777 */                 //     ^
/*   778 */                 //
/*   779 */                 currLineNo = 13;
/*   780 */                 currColNo = 4;
/*   781 */ 
/*   782 */                 if (e === undefined) {
/*   783 */                     throw new Sk.builtin.UnboundLocalError('local variable \'e\' referenced before assignment');
/*   784 */                 }
/*   785 */                 var $elem26 = e;
/*   786 */                 if (f === undefined) {
/*   787 */                     throw new Sk.builtin.UnboundLocalError('local variable \'f\' referenced before assignment');
/*   788 */                 }
/*   789 */                 var $elem27 = f;
/*   790 */                 var $loadtuple28 = new Sk.builtins['tuple']([$elem26, $elem27]);
/*   791 */                 Sk.misceval.print_(new Sk.builtins['str']($loadtuple28).v);
/*   792 */                 Sk.misceval.print_("\n");
/*   793 */                 if (Sk.breakpoints('<stdin>.py', 14, 4)) {
/*   794 */                     var $susp = $saveSuspension({
/*   795 */                         data: {
/*   796 */                             type: 'Sk.debug'
/*   797 */                         },
/*   798 */                         resume: function() {}
/*   799 */                     }, '<stdin>.py', 14, 4);
/*   800 */                     $susp.$blk = 4;
/*   801 */                     $susp.optional = true;
/*   802 */                     return $susp;
/*   803 */                 }
/*   804 */                 $blk = 4; /* allowing case fallthrough */
/*   805 */             case 4:
/*   806 */                 /* --- debug breakpoint for line 14 --- */
/*   807 */                 //
/*   808 */                 // line 14:
/*   809 */                 //     print("boo")
/*   810 */                 //     ^
/*   811 */                 //
/*   812 */                 currLineNo = 14;
/*   813 */                 currColNo = 4;
/*   814 */ 
/*   815 */                 var $str29 = new Sk.builtins['str']('boo');
/*   816 */                 Sk.misceval.print_(new Sk.builtins['str']($str29).v);
/*   817 */                 Sk.misceval.print_("\n");
/*   818 */                 return Sk.builtin.none.none$;
/*   819 */                 throw new Sk.builtin.SystemError('internal error: unterminated block');
/*   820 */             }
/*   821 */         } catch (err) {
/*   822 */             if (!(err instanceof Sk.builtin.BaseException)) {
/*   823 */                 err = new Sk.builtin.ExternalError(err);
/*   824 */             }
/*   825 */             err.traceback.push({
/*   826 */                 lineno: currLineNo,
/*   827 */                 colno: currColNo,
/*   828 */                 filename: '<stdin>.py'
/*   829 */             });
/*   830 */             if ($exc.length > 0) {
/*   831 */                 $err = err;
/*   832 */                 $blk = $exc.pop();
/*   833 */                 continue;
/*   834 */             } else {
/*   835 */                 throw err;
/*   836 */             }
/*   837 */         }
/*   838 */     }
/*   839 */ });
```



## Diff of what's extra:
![Debug Code](/img/presentation/skulpt/skulpt_diff_1.png "Skulpt Diff")



## Debugger Structure
- 2 files
- Debugger
- Debugger frontend for CodeMirror



## Debugger Execution
```javascript
        Sk.configure({
            output: repl.print,
            debugout: window.jsoutf,
            read: window.builtinRead,
            yieldLimit: null,
            execLimit: null,
        });
        
        try {
            var susp_handlers = {};
            susp_handlers["*"] = repl.sk_debugger.suspension_handler.bind(this);
            Sk.breakpoints = repl.sk_debugger.check_breakpoints.bind(repl.sk_debugger);
            
            var promise = repl.sk_debugger.asyncToPromise(function() {
                return Sk.importMainWithBody(editor_filename, true, repl.sk_code_editor.getValue(),true);
            }, susp_handlers, this.sk_debugger);
            promise.then(this.sk_debugger.success.bind(this.sk_debugger), this.sk_debugger.error.bind(this.sk_debugger));
        } catch(e) {
            outf(e.toString() + "\n")
```



## On Success
```javascript
Sk.Debugger.prototype.success = function(r) {
    if (r instanceof Sk.misceval.Suspension) {
        this.set_suspension(r);
    }
};
```



## Catching Errors
```javascript
Sk.Debugger.prototype.error = function(e) {
    this.print("Traceback (most recent call last):");
    for (var idx = 0; idx < e.traceback.length; ++idx) {
        this.print("  File \"" + e.traceback[idx].filename + "\", line " + e.traceback[idx].lineno + ", in <module>");
        var code = this.get_source_line(e.traceback[idx].lineno - 1);
        code = code.trim();
        code = "    " + code;
        this.print(code);
    }
    
    var err_ty = e.constructor.tp$name;
    for (var idx = 0; idx < e.args.v.length; ++idx) {
        this.print(err_ty + ": " + e.args.v[idx].v);
    }
};
```



## On Resume
```javascript
Sk.Debugger.prototype.suspension_handler = function(susp) {
    return new Promise(function(resolve, reject) {
        try {
            resolve(susp.resume());
        } catch(e) {
            reject(e);
        }
    });
};

Sk.Debugger.prototype.resume = function() {
    // Reset the suspension stack to the topmost
    this.current_suspension = this.suspension_stack.length - 1;
    
    if (this.suspension_stack.length == 0) {
        this.print("No running program");
    } else {
        var promise = this.suspension_handler(this.get_active_suspension());
        promise.then(this.success.bind(this), this.error.bind(this));
    }
};
```



## Recursive Call stack Resolution
```javascript
Sk.Debugger.prototype.success = function(r) {
    if (r instanceof Sk.misceval.Suspension) {
        this.set_suspension(r);
    } else {
        if (this.suspension_stack.length > 0) {
            // Current suspension needs to be popped of the stack
            this.suspension_stack.pop();
            
            if (this.suspension_stack.length == 0) {
                this.print("Program execution complete");
                return;
            }
            
            var parent_suspension = this.get_active_suspension();
            // The child has completed the execution. So override the child's resume
            // so we can continue the execution.
            parent_suspension.child.resume = function() {};  // <------ Very Important line

            this.print_suspension_info(parent_suspension);
        } else {
            this.print("Program execution complete");
        }
    }
};
```



## Implemented Features
- Print current stack trace
- Move up / down the stack
- inspect variables at a current stack
- setting & clearing of breakpoints
- setting of temporary breakpoints
- enable / disable breakpoints
- ignore breakpoint for certain count
- Step Over / Step in
- continue execution (partial)
- list source around current execution point



## Future Features
- Conditional Breakpoints
- True Step in feature
- Changing variable values
- Handling yields
- Integration with CodeMirror



## That's about it.. Questions