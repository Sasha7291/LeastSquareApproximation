#pragma once

#include "lsa_linearsystemsolver.hpp"

#include <numeric>
#include <vector>

#include <QDebug>
#include <QVector>


namespace lsa
{

using Keys = std::span<const double>;
using Values = std::span<const double>;
template<const std::size_t N>
using Coefficients = std::array<double, N>;
using ResultValues = std::vector<double>;
template<const std::size_t N>
using Result = std::pair<Coefficients<N>, ResultValues>;

class Approximator
{

public:
    Approximator() noexcept = default;
    ~Approximator() noexcept = default;

    Approximator(const Approximator &) = delete;
    Approximator(Approximator &&) = delete;
    Approximator &operator=(const Approximator &) = delete;
    Approximator &operator=(Approximator &&) = delete;

    [[nodiscard]] Result<2> exponential(Keys x, Values y) const noexcept;
    [[nodiscard]] Result<2> linear(Keys x, Values y) const noexcept;
    template<const std::size_t N>
    [[nodiscard]] Result<N> polynomial(Keys x, Values y) const noexcept;
	
private:
    template<const std::size_t N>
	[[nodiscard]] ResultValues approximateValues(const Coefficients<N> &coeffs, Keys x) const noexcept;

};

Result<2> Approximator::exponential(Keys x, Values y) const noexcept
{
    auto minValue = *std::ranges::min_element(y);
    ResultValues tempY(y.size());
    std::ranges::transform(y, tempY.begin(), [minValue](double value) -> double {
        return std::log(value - minValue + 0.001);
    });
    auto [coeffs, result] = linear(x, tempY);
	coeffs[0] = std::exp(coeffs[0]);
    std::ranges::transform(result, result.begin(), [minValue](double value) -> double {
        return std::exp(value) + minValue - 0.001;
	});
	
    return std::make_pair(coeffs, result);
}

Result<2> Approximator::linear(Keys x, Values y) const noexcept
{
    return polynomial<2ull>(x, y);
}

template<const std::size_t N>
Result<N> Approximator::polynomial(Keys x, Values y) const noexcept
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

    auto result = LinearSystemSolver()(A, B).column(0);
    return std::make_pair(result, approximateValues(result, x));
}

template<const std::size_t N>
[[nodiscard]] ResultValues Approximator::approximateValues(const Coefficients<N> &coeffs, Keys x) const noexcept
{
    ResultValues r;
    r.reserve(x.size());

    for (auto i = 0ull; i < x.size(); ++i)
    {
        double sum = 0.0;

        for (auto j = 0; j < N; ++j)
            sum += std::pow(x[i], j) * coeffs[j];

        r.push_back(sum);
    }
	
	return r;
}

}
