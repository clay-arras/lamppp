#include "grad.h"
#include "engine.h"

void add_backward(void* ctx) {
  auto* context = static_cast<AddBackwardContext*>(ctx);
  context->self->grad += 1.0 * context->out->grad;
  context->other->grad += 1.0 * context->out->grad;
  delete context;
}

void mul_backward(void* ctx) {
  auto* context = static_cast<MulBackwardContext*>(ctx);
  context->self->grad += context->other->data * context->out->grad;
  context->other->grad += context->self->data * context->out->grad;
  delete context;
}

void pow_backward(void* ctx) {
  auto* context = static_cast<PowBackwardContext*>(ctx);
  context->self->grad +=
      context->other->data *
      std::pow(context->self->data, context->other->data - 1) *
      context->out->grad;
  // context->other->grad += std::log(context->self->data) *
  //     std::pow(context->self->data, context->other->data) *
  //     context->out->grad;
  delete context;
}

void exp_backward(void* ctx) {
  auto* context = static_cast<ExpBackwardContext*>(ctx);
  context->self->grad += context->out->data * context->out->grad;
  delete context;
}

void log_backward(void* ctx) {
  auto* context = static_cast<LogBackwardContext*>(ctx);
  context->self->grad += (1.0 / context->self->data) * context->out->grad;
  delete context;
}

void relu_backward(void* ctx) {
  auto* context = static_cast<ReluBackwardContext*>(ctx);
  context->self->grad += (context->self->data > 0) * context->out->grad;
  delete context;
}
