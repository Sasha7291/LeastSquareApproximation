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
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept;
    [[nodiscard]] Type pearsonCoefficient(Keys x, Values y) const noexcept;
    [[nodiscard]] ResultValues rank(Keys data) const noexcept;
    [[nodiscard]] Type spearmanCoefficient(Keys x, Values y) const noexcept;
    [[nodiscard]] ResultValues standardize(Keys data) const noexcept;
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

Coefficients Statistics::coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept
{
    const auto aver = average(data);
    const auto var = variance(data);
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
            result[i] += coeffs[j] * binomialCoefficient(j, i) * std::pow(-aver, j - i) / std::pow(var, j);

    return result;
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

ResultValues Statistics::standardize(Keys data) const noexcept
{
    ResultValues result(data.size());

    std::ranges::transform(data, result.begin(), [aver = average(data), var = variance(data)](Type value) -> Type {
        return (value - aver) / var;
    });

    return result;
}

Type Statistics::variance(Keys data) const noexcept
{
    return std::sqrt(std::transform_reduce(data.cbegin(), data.cend(), 0.0, std::plus<>(), [aver = average(data)](Type val) -> Type {
		return (val - aver) * (val - aver);
	}) / data.size());
}

}
