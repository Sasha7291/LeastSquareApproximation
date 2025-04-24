#pragma once

#include "squarematrix.hpp"

#ifdef QT
#include <QSpan>
#include <QVector>
#else
#include <span>
#include <vector>
#endif

#include <optional>


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
using Result = Coefficients;
#else
using Keys = std::span<const Type>;
using Values = std::span<const Type>;
using Coefficients = std::vector<Type>;
using ResultValues = std::vector<Type>;
using Result = std::pair<Coefficients, ResultValues>;
#endif
}
