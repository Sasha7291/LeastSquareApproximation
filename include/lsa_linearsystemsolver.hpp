#pragma once

#include <squarematrix.hpp>


namespace lsa
{
	
class LinearSystemSolverException : std::runtime_error
{
public:
    LinearSystemSolverException(const std::string &message) : std::runtime_error(message) {}
};


class LinearSystemSolver
{
public:
    LinearSystemSolver() noexcept = default;
    ~LinearSystemSolver() noexcept = default;

    LinearSystemSolver(const LinearSystemSolver &) = delete;
    LinearSystemSolver(LinearSystemSolver &&) = delete;
    LinearSystemSolver &operator=(const LinearSystemSolver &) = delete;
    LinearSystemSolver &operator=(LinearSystemSolver &&) = delete;

    template<NumberType T, const std::size_t N>
    [[nodiscard]] Matrix::Matrix<T, N, 1> operator()(
        const Matrix::SquareMatrix<T, N> &A,
        const Matrix::Matrix<T, N, 1> &B
    ) const;

};

template<NumberType T, const std::size_t N>
Matrix::Matrix<T, N, 1> LinearSystemSolver::operator()(
    const Matrix::SquareMatrix<T, N> &A,
    const Matrix::Matrix<T, N, 1> &B
) const
{
    if (std::abs(A.determinant()) <= std::numeric_limits<T>::epsilon())
        throw LinearSystemSolverException("det(A) == 0, A is irreversible");

    return A.inverted().toMatrix() * B;
}
	
}
