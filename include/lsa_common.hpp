#pragma once

#include "squarematrix.hpp"

#ifdef QT
#include <QSpan>
#include <QVector>
#else
#include <span>
#include <vector>
#endif


namespace lsa 
{

template<NumberType T>
using CoefficientMatrix = dynamic_matrix::SquareMatrix<T>;
template<NumberType T>
using FreeMemberColumn = dynamic_matrix::Matrix<T>;
template<NumberType T>
using UnknownColumn = dynamic_matrix::Matrix<T>;

using Type = double;


class Exception : public std::runtime_error
{
public:
    Exception(const std::string &message) : std::runtime_error(message) {}
};


#ifdef QT
using Keys = QSpan<const Type>;
using Values = QSpan<const Type>;
using Coefficients = QVector<Type>;
using ResultValues = QVector<Type>;
using Result = QPair<Coefficients, ResultValues>;
#else
using Keys = std::span<const Value>;
using Values = std::span<const Value>;
using Coefficients = std::vector<Value>;
using ResultValues = std::vector<Value>;
using Result = std::pair<Coefficients, ResultValues>;
#endif

}
