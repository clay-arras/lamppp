LazyNode
- shared_ptr<LazyFunction>
- shared_ptr<TensorImpl> # leaf node

shape, device, and dtype
The root node is a TensorImpl that stores a LazyFunction
lazyFunction holds lists of lazyNodes


