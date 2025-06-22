#include "lamppp/nets/any.hpp"

namespace lmp::nets {

std::shared_ptr<ModuleImpl> AnyModule::getImpl() { return impl_->getImpl(); };

std::any AnyModule::call(const std::vector<std::any>& args) const {
return impl_->call(args);
}

}