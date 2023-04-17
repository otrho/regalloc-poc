# Proof Of Concept Register Allocation Using `bytecodealliance/regalloc2`

This is a little example of how to implement register allocation on a tiny toy language using
[`bytecodealliance/regalloc2`](https://github.com/bytecodealliance/regalloc2) (the new register allocation crate for CraneLift).

It uses a tiny expression language which is converted to IR and then to a simple virtual ASM.  The
IR and first stage abstract ASM use an infinite register pool and we then use `regalloc2` to
allocate concrete registers.

By default a completely random expression is generated per run.

At each stage (expr, IR, abstract ASM, ASM) we have an interpreter which can evaluate the
expression. The program 'succeeds' if each of these evaluations coincide.

## Example

Running the program might produce something like the following example.

### The expression in prefix s-exp notation

Allows conditionals and addition of integers. Semantically, the conditionals evaluate their second
arg if the first arg is non-zero, otherwise the first arg.  In practice, due to the unlikeliness of
randomly generating an expression equalling zero as the first arg, the second arg is almost always
evaluated.  But this is beside the point.

```
(+ (+ 1 2) (if (+ 3 4) 5 (+ 6 7)))
```

### The IR

The point of adding conditionals is so the IR has some control flow and the need for basic blocks.
The blocks take args instead of using PHI instructions.

```
block0():
    %2 = add 1 2
    %5 = add 3 4
    cbr %5 block1() block2()

block1():
    br block3(5)

block2():
    %9 = add 6 7
    br block3(%9)

block3(%11):
    %14 = add %2 %11
    ret %14
```

### The abstract ASM

The abstract ASM uses unlimited registers, numbered starting at 1000 to make it visually distinct
from the allocated ASM.  This is obviously unoptimised but that is also beside the point.

```
  0 label_0:
  1     mov %1001 1
  2     mov %1002 2
  3     add %1000 %1001 %1002
  4     mov %1004 3
  5     mov %1005 4
  6     add %1003 %1004 %1005
  7     jz %1003 label_1
  8     jmp label_2
  9 label_2:
 10     mov %1009 5
 11     mov %1008 %1009
 12     jmp label_3
 13 label_1:
 14     mov %1011 6
 15     mov %1012 7
 16     add %1010 %1011 %1012
 17     mov %1008 %1010
 18     jmp label_3
 19 label_3:
 20     add %1014 %1000 %1008
 21     ret %1014
```

### The allocated ASM

The ASM using a constrained register pool.  In this case we have only 4 registers, numbered `%16` to
`%19`.  Note that the allocator removed some redundant `move` instructions (lines 11 and 17 above).

```
  0 label_0:
  1     mov %18 1
  2     mov %16 2
  3     add %19 %18 %16
  4     mov %16 3
  5     mov %18 4
  6     add %17 %16 %18
  7     jz %17 label_1
  8     jmp label_2
  9 label_2:
 10     mov %17 5
 11     jmp label_3
 12 label_1:
 13     mov %16 6
 14     mov %17 7
 15     add %17 %16 %17
 16     jmp label_3
 17 label_3:
 18     add %19 %19 %17
 19     ret %19
```

### The evaluation summary

At each stage the representation is evaluated, and in this case they all resolve to 8, which is a
successful run.

```
EXPR RESULT: 8
SSA RESULT: 8
ABSTRACT ASM RESULT: 8
ALLOC'D ASM RESULT: 8

SUCCESS
```

## Comments

Using the `regalloc2` crate is fairly straight forward.  The hardest part is implementing the
`Function` trait for the abstract ASM, as it tends to assume certain data structures and data
layouts.

It would prefer the input to be SSA though this isn't necessary, and the ASM input provided in this
PoC is not SSA.  It requires the input be structured as basic blocks, much like the IR, but at the
same time assumes the instructions in a function can be easily indexed, as if they're in an array.
This is the layout used in this PoC.

To keep things vaguely efficient a bunch of metadata is gathered before allocation, such as the
block boundaries, their predecessors and successors and the instruction operand use/def
characteristics.

The opcodes are all then rewritten into a new array.  This is a bit wasteful but due to spilling
it's possible that new instructions (load and stores mostly) need to be inserted into the ASM and so
updating the existing opcodes, by inserting into a `Vec` would also be quite inefficient.  The
rewriting is done per block though, so it may be possible to save RAM use by dropping the old blocks
as they are rewritten.  This would require them not to all be in a single array.

### Spilling

One of the trickiest parts of register allocation is implementing spilling, and this is one of the
driving reasons to use this crate rather than rolling our own.  By declaring every instruction to
use or define registers only, we force the allocator to never necessitate reading or writing
directly to the stack.  Instead it provides 'edits' which indicate where loads and stores should be
made.

The PoC will generate expressions randomly, often they're quite small and don't need to spill, but
sometimes they're quite large.  To make the need to spill (and provide coverage for the PoC) the
number of registers allowed is only 4.  Usually it only takes a run or two for these to turn up, and
they can be found by just searching the output for `load` or `store` which are never emitted by the
abstract assembler.

