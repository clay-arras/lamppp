

<!-- - decision: do I keep the unique_ptr OR do I do shared ptr? 
- default behavior: do I want Modules to be COPIABLE, or one-to-many PIMPL type deal? -->

- do I want maps to AnyModule (type erased interface) OR to a shared_ptr<ModuleImpl>???
- so we DO need a ptr() function that returns a pointer (based on ModuleImpl) to the actual type


- so we just have a virtual function in Placeholder, concrete impl in Holder, and a interface method in AnyModule
