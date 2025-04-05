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

vector and unordered_set are pretty much the same

Nevermind;
IMPORTANT NOTE: THERE SILL IS REFERENCES TO THAT VALUE, NEED TO FIX THE MAKE_SHARED THING!!!!!!!!!
step 1: fix the references and figure out some way to calculate WITH THE OTHER VALUE BEING A CONSTANT
ok so if you have a graph with all the values being no grad, then the IN BETWEEN VALUES will be deleted; i.e. they'll be deallocated once the time's up
if the operation involves one requires_grad and one no_grad, then we still need to keep that no_grad reference, and there's no way around it.
