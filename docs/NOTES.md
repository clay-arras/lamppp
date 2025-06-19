

### changes: 
- step 1: ONLY require the user to implement a Impl class, NOT both (make the second one a macro)
- step 2: make the ModuleCRTP include the concrete Impl class (move it from general Module, since it makes the concrete forward signature muddly)
- step 3: extra -- add an auto-register macro
- step 4: implement the auto-constructor template creators
- step 5: merge Module and ModuleCRTP

### workflow sketch:
- Implement LinearImpl (doesn't inherit from anything, but NEEDS forward due to CRTP definition from ModuleCRTP) -- WRONG. LinearImpl inherits from Module!!!


### questions: 
- how do we define the constructor of the Interface class if we have the Impl class?



### maps:

Lamp++ : Libtorch
ModuleCRTP -> ModuleHolder
ModuleImpl -> Module
