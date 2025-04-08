#pragma once

#include "squarematrix.hpp"


namespace lsa
{
	
class Exception : std::runtime_error
{
public:
    Exception(const std::string &message) : std::runtime_error(message) {}
};


class Solver
{
public:
    Solver() noexcept = default;
    ~Solver() noexcept = default;

    Solver(const Solver &) = delete;
    Solver(Solver &&) = delete;
    Solver &operator=(const Solver &) = delete;
    Solver &operator=(Solver &&) = delete;

    template<NumberType T>
    [[nodiscard]] dynamic_matrix::Matrix<T> operator()(
        const dynamic_matrix::SquareMatrix<T> &A,
        const dynamic_matrix::Matrix<T> &B
    ) const;

};

template<NumberType T>
dynamic_matrix::Matrix<T> Solver::operator()(
    const dynamic_matrix::SquareMatrix<T> &A,
    const dynamic_matrix::Matrix<T> &B
) const
{
    if (std::abs(A.determinant()) <= std::numeric_limits<T>::epsilon())
        throw Exception("det(A) == 0, A is irreversible");

    return A.inverted().toMatrix() * B;
}
	
}
