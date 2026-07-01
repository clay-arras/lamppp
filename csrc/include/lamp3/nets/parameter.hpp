#pragma once

#include "lamp3/autograd/variable.hpp"

namespace lmp::nets {

class Parameter : public autograd::Variable {}; // literally just Variable

}