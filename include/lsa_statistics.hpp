#pragma once

#include "lsa_common.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>


namespace lsa
{

class Statistics
{

public:
	Statistics() noexcept = default;
	~Statistics() noexcept = default;
	
	Statistics(const Statistics &) = delete;
    Statistics(Statistics &&) = delete;
    Statistics &operator=(const Statistics &) = delete;
    Statistics &operator=(Statistics &&) = delete;

    [[nodiscard]] ResultValues operator()(Keys x, Values y) const noexcept;

    [[nodiscard]] Type average(Keys data) const noexcept;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const noexcept;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept;
    [[nodiscard]] Type pearsonCoefficient(Keys x, Values y) const noexcept;
    [[nodiscard]] ResultValues rank(Keys data) const noexcept;
    [[nodiscard]] Type spearmanCoefficient(Keys x, Values y) const noexcept;
    [[nodiscard]] ResultValues standardize(Keys data, Type average, Type variance) const noexcept;
    [[nodiscard]] ResultValues standardize(Keys data) const noexcept;
    [[nodiscard]] Type variance(Keys data, Type average) const noexcept;
    [[nodiscard]] Type variance(Keys data) const noexcept;
	
};

ResultValues Statistics::operator ()(Keys x, Values y) const noexcept
{
    return {
        pearsonCoefficient(x, y),
        spearmanCoefficient(x, y)
    };
}

Type Statistics::average(Keys data) const noexcept
{
    return std::accumulate(data.cbegin(), data.cend(), 0.0) / data.size();
}

Coefficients Statistics::coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const noexcept
{
    const auto n = coeffs.size();
    Coefficients result(n, 0);

    const std::function<int(int)> factorial = [&factorial](int k) -> int {
        return (k > 1) ? k * factorial(k - 1) : 1;
    };
    const std::function<int(int, int)> binomialCoefficient = [&factorial](int n, int k) -> int {
        return factorial(n) / (factorial(k) * factorial(n - k));
    };

    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i; j < n; ++j)
            result[i] += coeffs[j] * binomialCoefficient(j, i) * std::pow(-average, j - i) / std::pow(variance, j);

    return result;
}

Coefficients Statistics::coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept
{
    auto aver = average(data);
    return coefficientReverseStandardization(coeffs, aver, variance(data, aver));
}

Type Statistics::pearsonCoefficient(Keys x, Values y) const noexcept
{
    return std::transform_reduce(x.cbegin(), x.cend(), y.cbegin(), 0.0, std::plus<>(), [xAverage = average(x), yAverage = average(y)](Type val1, Type val2) -> Type {
		return (val1 - xAverage) * (val2 - yAverage);
    }) / (x.size() * variance(x) * variance(y));
}

ResultValues Statistics::rank(Keys data) const noexcept
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

Type Statistics::spearmanCoefficient(Keys x, Values y) const noexcept
{
    auto rankX = rank(x);
    auto rankY = rank(y);

    return 1.0 - 6.0 * std::transform_reduce(rankX.cbegin(), rankX.cend(), rankY.cbegin(), 0.0, std::plus<>(), [](Type rank1, Type rank2) -> Type {
		return (rank1 - rank2) * (rank1 - rank2);
	}) / (x.size() * (x.size() * x.size() - 1));
}

ResultValues Statistics::standardize(Keys data, Type average, Type variance) const noexcept
{
    ResultValues result(data.size());

    std::ranges::transform(data, result.begin(), [average, variance](Type value) -> Type {
        return (value - average) / variance;
    });

    return result;
}

ResultValues Statistics::standardize(Keys data) const noexcept
{
    auto aver = average(data);
    return standardize(data, aver, variance(data, aver));
}

Type Statistics::variance(Keys data, Type average) const noexcept
{
    return std::sqrt(std::transform_reduce(data.cbegin(), data.cend(), 0.0, std::plus<>(), [average](Type val) -> Type {
        return (val - average) * (val - average);
    }) / data.size());
}

Type Statistics::variance(Keys data) const noexcept
{
    return variance(data, average(data));
}

}
