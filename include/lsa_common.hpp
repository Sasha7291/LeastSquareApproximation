#pragma once

#include "squarematrix.hpp"


namespace lsa 
{

template<NumberType T>
using CoefficientMatrix = dynamic_matrix::SquareMatrix<T>;
template<NumberType T>
using FreeMemberColumn = dynamic_matrix::Matrix<T>;
template<NumberType T>
using UnknownColumn = dynamic_matrix::Matrix<T>;

using Keys = std::span<const double>;
using Values = std::span<const double>;
using Coefficients = std::vector<double>;
using ResultValues = std::vector<double>;
using Result = std::pair<Coefficients, ResultValues>;

}
