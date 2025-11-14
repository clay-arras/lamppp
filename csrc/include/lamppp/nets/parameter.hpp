#pragma once

#include "lamppp/autograd/variable.hpp"

namespace lmp::nets {

class Parameter : public autograd::Variable {}; // literally just Variable

}