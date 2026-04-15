#pragma once

#include "lsa_common.hpp"


namespace lsa
{

class Solver
{

public:
    Solver() = default;
    ~Solver() = default;

    Solver(const Solver &) = delete;
    Solver(Solver &&) = delete;
    Solver &operator=(const Solver &) = delete;
    Solver &operator=(Solver &&) = delete;

    template<NumberType T>
    [[nodiscard]] std::optional<UnknownColumn<T>> operator()(
        const CoefficientMatrix<T> &A,
        const FreeMemberColumn<T> &B
    ) const;

};

template<NumberType T>
std::optional<UnknownColumn<T>> Solver::operator()(
    const CoefficientMatrix<T> &A,
    const FreeMemberColumn<T> &B
) const
{
    std::size_t n = A.columnCount();
    UnknownColumn<T> X(n, 1);

    // Создаём расширенную матрицу [A | B]
    std::vector<std::vector<T>> aug(n, std::vector<T>(n + 1));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) aug[i][j] = A(i, j);
        aug[i][n] = B(i, 0);
    }

    // Прямой ход Гаусса
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t pivot = i;
        for (std::size_t k = i + 1; k < n; ++k)
            if (std::abs(aug[k][i]) > std::abs(aug[pivot][i])) pivot = k;
        std::swap(aug[i], aug[pivot]);
        if (std::abs(aug[i][i]) < 1e-12) return std::nullopt; // Сингулярная матрица

        for (std::size_t k = i + 1; k < n; ++k) {
            T factor = aug[k][i] / aug[i][i];
            for (std::size_t j = i; j <= n; ++j)
                aug[k][j] -= factor * aug[i][j];
        }
    }

    // Обратный ход
    for (std::size_t i = 0; i < n; ++i) X(i, 0) = aug[i][n];
    for (int i = n - 1; i >= 0; --i) {
        for (std::size_t j = i + 1; j < n; ++j)
            X(i, 0) -= aug[i][j] * X(j, 0);
        X(i, 0) /= aug[i][i];
    }

    return X;
}
	
}
