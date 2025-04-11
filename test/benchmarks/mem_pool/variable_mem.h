#ifndef _VARIABLE_MEM_H_
#define _VARIABLE_MEM_H_

#include "autograd/engine/function.h"
#include "value_pool.h"
#include "variable.h"

class VariableMem : public Variable {
 public:
  static ValueMemoryPool pool_;
  static void destroy(VariableImpl* ptr);

  VariableMem();
  explicit VariableMem(float data);
};

#endif  // _VARIABLE_MEM_H_