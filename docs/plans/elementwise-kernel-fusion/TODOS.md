

should it be data() from TensorImpl or Tensor which realizes the backend
compilation? if its TensorImpl, you get a bunch of places in dispatch
handlers that call it (but thats eager path, so maybe it doesn't
matter)? is there a case where this can get very messy?

but intuitively, TensorImpl is the dedicated business logic point, and
it should be routed as so (Tensor is light interface)


---
TODO: need end to end tests that tests the entire workflow and complex autograd features.

need a CLAUDE skill that mirrors claude --resume
af3da953-e5f4-425d-8399-6d0db0396eb3 workflow. spawns subagents, review
PR, also should commit and push automatically. create a worker agent


BTW: binary vs broadcast: flag on the base
this is a sus decision, but if we need to pivot (indexing logic doesn't
work out), it should be a localized refactor
