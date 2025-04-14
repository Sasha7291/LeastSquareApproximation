#pragma once

#include "lsa_solver.hpp"
#include "lsa_statistics.hpp"

#include <algorithm>
#include <numeric>


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

    [[nodiscard]] Result exponential(Keys x, Values y) const;
    [[nodiscard]] Result linear(Keys x, Values y) const;
    [[nodiscard]] Result polynomial(Keys x, Values y, const std::size_t N) const ;
	
private:
    [[nodiscard]] ResultValues approximateValues(const Coefficients &coeffs, Keys x) const noexcept;

};

Result Approximator::exponential(Keys x, Values y) const
{
    auto minValue = *std::ranges::min_element(y);
    ResultValues tempY(y.size());
    std::ranges::transform(y, tempY.begin(), [minValue](Type value) -> Type {
        return std::log(value - minValue + 0.001);
    });

    try
    {
        auto [coeffs, result] = linear(x, tempY);
        coeffs[0] = std::exp(coeffs[0]);
        std::ranges::transform(result, result.begin(), [minValue](Type value) -> Type {
            return std::exp(value) + minValue - 0.001;
        });

        return std::make_pair(coeffs, result);
    }
    catch (const Exception &exception)
    {
        throw exception;
    }
}

Result Approximator::linear(Keys x, Values y) const
{
    try
    {
        return polynomial(x, y, 2ull);
    }
    catch (const Exception &exception)
    {
        throw exception;
    }
}

Result Approximator::polynomial(Keys x, Values y, const std::size_t N) const
{
    Statistics statistics;
    auto aver = statistics.average(x);
    auto var = statistics.variance(x, aver);
    auto tempX = statistics.standardize(x, aver, var);

    dynamic_matrix::SquareMatrix<Type> A(N);
    dynamic_matrix::Matrix<Type> B(N, 1);

    ResultValues sums(2 * N - 1);
    for (auto i = 0ull; i < sums.size(); ++i)
        sums[i] = std::accumulate(tempX.cbegin(), tempX.cend(), 0.0, [i](Type total, Type value) -> Type {
            return total + std::pow(value, i);
        });

    for (auto i = 0ull; i < N; ++i)
    {
        for (auto j = i; j < N; ++j)
            A(i, j) = A(j, i) = sums[i + j];

        B(i, 0) = std::transform_reduce(tempX.cbegin(), tempX.cend(), y.cbegin(), 0.0, std::plus<>(), [i](Type value1, Type value2) -> Type {
            return std::pow(value1, i) * value2;
        });
    }

    auto unknownColumn = Solver()(A, B);
    if (!unknownColumn.has_value())
        throw Exception("Coefficients matrix is irreversible");

    auto coeffs = statistics.coefficientReverseStandardization(unknownColumn.value().column(0), aver, var);
    return std::make_pair(coeffs, approximateValues(coeffs, x));
}

[[nodiscard]] ResultValues Approximator::approximateValues(const Coefficients &coeffs, Keys x) const noexcept
{
    ResultValues r(x.size(), static_cast<double>(0));
    r.reserve(x.size());

    for (auto i = 0ull; i < x.size(); ++i)
        for (auto j = 0; j < coeffs.size(); ++j)
            r[i] += std::pow(x[i], j) * coeffs[j];
	
	return r;
}

}
