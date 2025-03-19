#pragma once

#include <numeric>
#include "linearsystemsolver.hpp"


namespace lsa
{
	
class Approximator
{

public:
    Approximator() noexcept = default;
    ~Approximator() noexcept = default;

    Approximator(const Approximator &) = delete;
    Approximator(Approximator &&) = delete;
    Approximator &operator=(const Approximator &) = delete;
    Approximator &operator=(Approximator &&) = delete;

    [[nodiscard]] std::array<double, 2ull> exponential(std::span<double> x, std::span<double> y);
    [[nodiscard]] std::array<double, 2ull> linear(std::span<double> x, std::span<double> y);
    template<const std::size_t N>
    [[nodiscard]] std::array<double, N> polynomial(std::span<double> x, std::span<double> y);

};

std::array<double, 2ull> Approximator::exponential(std::span<double> x, std::span<double> y)
{
    std::ranges::transform(y, y.begin(), [](double value) -> double {
        return std::log(value);
    });
    auto result = linear(x, y);
    return std::array<double, 2>({ std::exp(result[0]), result[1] });
}

std::array<double, 2ull> Approximator::linear(std::span<double> x, std::span<double> y)
{
    return polynomial<2ull>(x, y);
}

template<const std::size_t N>
std::array<double, N> Approximator::polynomial(std::span<double> x, std::span<double> y)
{
    Matrix::SquareMatrix<double, N> A;
    Matrix::Matrix<double, N, 1> B;

    for (auto i = 0ull; i < N; ++i)
    {
        for (auto j = 0ull; j < N; ++j)
            A(i, j) = std::accumulate(x.begin(), x.end(), 0.0, [i, j](double total, double value) -> double {
                return total + std::pow(value, i + j);
            });

        B(i, 0) = std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<>(), [i](double value1, double value2) -> double {
            return std::pow(value1, i) * value2;
        });
    }

    LinearSystemSolver solver;
    return solver(A, B).column(0);
}

}
