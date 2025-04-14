#pragma once

#include "lsa_common.hpp"


namespace lsa
{

class Arithmetic
{

public:
    Arithmetic() noexcept = default;
    ~Arithmetic() noexcept = default;

    Arithmetic(const Arithmetic &) = delete;
    Arithmetic(Arithmetic &&) = delete;
    Arithmetic &operator=(const Arithmetic &) = delete;
    Arithmetic &operator=(Arithmetic &&) = delete;

    [[nodiscard]] Type average(Keys data) const noexcept;
    [[nodiscard]] constexpr unsigned binomialCoefficient(unsigned n, unsigned k) const noexcept;
    [[nodiscard]] constexpr unsigned factorial(unsigned n) const noexcept;
    [[nodiscard]] Type integrate(Keys x, Values y) const noexcept;
    [[nodiscard]] Type powerArray(Keys data, unsigned degree) const noexcept;
    [[nodiscard]] ResultValues rank(Keys data) const noexcept;
    [[nodiscard]] ResultValues standardize(Keys data, Type average, Type variance) const noexcept;
    [[nodiscard]] ResultValues standardize(Keys data) const noexcept;
    [[nodiscard]] Type variance(Keys data, Type average) const noexcept;
    [[nodiscard]] Type variance(Keys data) const noexcept;

};


Type Arithmetic::average(Keys data) const noexcept
{
    return std::accumulate(data.cbegin(), data.cend(), 0.0) / data.size();
}

constexpr unsigned Arithmetic::binomialCoefficient(unsigned n, unsigned k) const noexcept
{
    return factorial(n) / (factorial(k) * factorial(n - k));
}

constexpr unsigned Arithmetic::factorial(unsigned n) const noexcept
{
    return (n > 1) ? n * factorial(n - 1) : 1;
}

Type Arithmetic::integrate(Keys x, Values y) const noexcept
{
    Type result = static_cast<Type>(0);

    for (auto i = 1; i < x.size() - 1; ++i)
        result += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);

    return result;
}

Type Arithmetic::powerArray(Keys data, unsigned degree) const noexcept
{
    return std::accumulate(data.cbegin(), data.cend(), 0.0, [degree](Type total, Type value) -> Type {
        return total + std::pow(value, degree);
    });
}

ResultValues Arithmetic::rank(Keys data) const noexcept
{
    ResultValues rankData(data.size());
    auto min = *std::ranges::min_element(data);

    for (auto i = 0ull; i < data.size(); ++i)
    {
        auto minIter = std::ranges::lower_bound(data, min);
        min = *minIter;
        rankData[minIter - data.begin()] = i;
    }

    return rankData;
}

ResultValues Arithmetic::standardize(Keys data, Type average, Type variance) const noexcept
{
    ResultValues result(data.size());

    std::ranges::transform(data, result.begin(), [average, variance](Type value) -> Type {
        return (value - average) / variance;
    });

    return result;
}

ResultValues Arithmetic::standardize(Keys data) const noexcept
{
    auto aver = average(data);
    return standardize(data, aver, variance(data, aver));
}

Type Arithmetic::variance(Keys data, Type average) const noexcept
{
    return std::sqrt(std::transform_reduce(data.cbegin(), data.cend(), 0.0, std::plus<>(), [average](Type val) -> Type {
        return (val - average) * (val - average);
    }) / data.size());
}

Type Arithmetic::variance(Keys data) const noexcept
{
    return variance(data, average(data));
}

}
