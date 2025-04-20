#pragma once

#include "lsa_solver.hpp"
#include "psr_average.h"
#include "psr_standartize.h"
#include "psr_variance.h"

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

    [[nodiscard]] Result linear(Keys x, Values y) const;
    [[nodiscard]] Result polynomial(Keys x, Values y, const std::size_t N) const ;
	
private:
    [[nodiscard]] ResultValues approximateValues(const Coefficients &coeffs, Keys x) const noexcept;
    [[nodiscard]] constexpr unsigned binomialCoefficient(unsigned n, unsigned k) const noexcept;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const noexcept;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept;
    [[nodiscard]] constexpr unsigned factorial(unsigned n) const noexcept;

};

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
    auto aver = psr::Average<double>{}(x);
    auto var = psr::Variance<double>{aver}(x);
    auto tempX = psr::Standartize<double>{aver, var}(x);

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

    auto coeffs = coefficientReverseStandardization(unknownColumn.value().column(0), aver, var);
    return std::make_pair(coeffs, approximateValues(coeffs, x));
}

ResultValues Approximator::approximateValues(const Coefficients &coeffs, Keys x) const noexcept
{
    ResultValues r(x.size(), static_cast<double>(0));
    r.reserve(x.size());

    for (auto i = 0ull; i < x.size(); ++i)
        for (auto j = 0; j < coeffs.size(); ++j)
            r[i] += std::pow(x[i], j) * coeffs[j];
	
	return r;
}

constexpr unsigned Approximator::binomialCoefficient(unsigned n, unsigned k) const noexcept
{
    return factorial(n) / (factorial(k) * factorial(n - k));
}

constexpr unsigned Approximator::factorial(unsigned n) const noexcept
{
    return (n > 1) ? n * factorial(n - 1) : 1;
}

Coefficients Approximator::coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const noexcept
{
    const auto n = coeffs.size();
    Coefficients result(n, 0);

    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i; j < n; ++j)
            result[i] += coeffs[j] * binomialCoefficient(j, i) * std::pow(-average, j - i) / std::pow(variance, j);

    return result;
}

Coefficients Approximator::coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept
{
    auto aver = psr::Average<double>{}(data);
    return coefficientReverseStandardization(coeffs, aver, psr::Variance<double>{aver}(data));
}

}
