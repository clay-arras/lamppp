use: https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html

- then make sure you add the shared_ptr as a custom type
- next step is to change the Layer Implementation to accept the new values

- need to add linting, comments, cleaner code, assertions, etc.
- integrating Eigen expression templates
- memory pool or custom allocator
- using manual pointer allocation
- replace lambda functions with static function pointers

documentation:

- creating many layers:

```cpp
    int nin = 108 * 108 * 3;
    Layer W1(nin, 1024);
    Layer W2(1024, 512);
    Layer W3(512, 256);
    Layer W4(256, 128);
    Layer W5(128, 10);
```

> FastLayer is 1.604, Layer is 13.399

IDEA: somehow CONDENSE THE GRAPH IF NO_GRAD, BECAUSE ALL THE OPERATIONS BEFORE GRAD CAN JUST BE INTO ONE

// before requires grad it took 700x slower
// after requires grad it was only 200x slower

no optimizations ..7000000000 ns
with requires_grad 2000000000 ns
with memory pool: .3000000000 ns

<!-- with mp + lambda optim -->

regular double ......10000000 ns

now, with local threads
.......232769086 ns 230000000 ns
regular, w/o memory pooling
.......232769086 ns 210000000 ns
.............240273 ns 200000 ns

.......244149687 ns 240000000 ns 3 // x400 slower
.............651494 ns 600000 ns 1100

regular
BM_MatrixMultiplicationSharedValue 240995770 ns 240867705 ns 3

with the shared value overhead stuff, double wrapper
BM_MatrixMultiplicationSharedFloat 137610098 ns 137565666 ns 5
BM_MatrixMultiplicationSharedValue 237733062 ns 237683481 ns 3
BM_MatrixMultiplicationSharedFloat 137566795 ns 137552101 ns 5

vector and unordered_set are pretty much the same

Nevermind;
IMPORTANT NOTE: THERE SILL IS REFERENCES TO THAT VALUE, NEED TO FIX THE MAKE_SHARED THING!!!!!!!!!
step 1: fix the references and figure out some way to calculate WITH THE OTHER VALUE BEING A CONSTANT
ok so if you have a graph with all the values being no grad, then the IN BETWEEN VALUES will be deleted; i.e. they'll be deallocated once the time's up
if the operation involves one requires_grad and one no_grad, then we still need to keep that no_grad reference, and there's no way around it.

test:
w/o unique ptr: 143015692 ns
w/ unique ptr and deepcopy: 1893248409 ns

---

steps;

- need to make it a struct
- variable.h
- function.h

BM_MatrixMultiplicationSharedValue 2943497714 ns
BM_MatrixMultiplicationFloat 1942519028 ns
BM_MatrixMultiplicationVariable 1262827408 ns
BM_MatrixMultiplicationDouble ....21922319 ns

## With Eigen

                                used to be       7000000000

BM_MatrixMultiplicationSharedValue 3031328406 ns 3030622607 ns 1
BM_MatrixMultiplicationFloat 1268677717 ns 1268332400 ns 1
BM_MatrixMultiplicationVariable 1214239516 ns 1213906649 ns 1
BM_MatrixMultiplicationDouble 21217777 ns 21213751 ns 33

## Regular Operations

BM_OperationsSharedValue 41761202 ns 41755884 ns 16
BM_OperationsFloat .......5831485 ns 5830558 ns 120
BM_OperationsVariable ....7192327 ns 7191369 ns 97
BM_OperationsDouble .......148388 ns 148355 ns 4795
BM_OperationsDoubleWrapper 445555 ns 445375 ns 1565

# TODOS

- to make it cleaner please delete the wrapper and make an internal struct instead
- look into Eigen tensors and switch from Matrix
- look into profiling software to figure out why it's so slow (`perf`)
- look into `cppcheck` static code checker
- look into other tensor libraries (move away from Eigen)
- refactor the codebase, make it cleaner, implement Variable (after baseline stuff)

- baseline the code using numpy??? (but its also a cpp library)
- note: learn about CPU types, cache misses, stuff like that
- benchmark memory pool and see the speedup from assigning with memory pool vs shared_ptr

questions:

- what is a good baseline for how fast the autograd engine SHOULD be?
- a) the speed of the variable ITSELF
- b) the speed of matrix computations
- also a decision: do I want to switch matrix providers or not

- also: do I want the variable to store a TENSOR or store a SINGLE VARIABLE
