#pragma once

#include "lsa_common.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>


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

private:
    [[nodiscard]] double average(Keys data) const noexcept;
    [[nodiscard]] double pearsonCoefficient(Keys x, Values y) const noexcept;
    [[nodiscard]] ResultValues rank(Keys data) const noexcept;
    [[nodiscard]] double spearmanCoefficient(Keys x, Values y) const noexcept;
    [[nodiscard]] double variance(Keys data) const noexcept;
	
};

ResultValues Statistics::operator ()(Keys x, Values y) const noexcept
{
    return {
        pearsonCoefficient(x, y),
        spearmanCoefficient(x, y)
    };
}

double Statistics::average(Keys data) const noexcept
{
	return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double Statistics::pearsonCoefficient(Keys x, Values y) const noexcept
{
	return std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<>(), [xAverage = average(x), yAverage = average(y)](double val1, double val2) -> double {
		return (val1 - xAverage) * (val2 - yAverage);
    }) / (x.size() * variance(x) * variance(y));
}

ResultValues Statistics::rank(Keys data) const noexcept
{
    ResultValues rankData(data.size());
	auto min = *std::ranges::min_element(data);
	
    for (auto i = 0ull; i < data.size(); ++i)
	{
		auto minIter = std::lower_bound(data.begin(), data.end(), min);
		min = *minIter;
		rankData[std::distance(minIter, data.begin())] = i;
	}
	
	return rankData;
}

double Statistics::spearmanCoefficient(Keys x, Values y) const noexcept
{
    auto rankX = rank(x);
    auto rankY = rank(y);

    return 1.0 - 6.0 * std::transform_reduce(rankX.cbegin(), rankX.cend(), rankY.cbegin(), 0.0, std::plus<>(), [](double rank1, double rank2) -> double {
		return (rank1 - rank2) * (rank1 - rank2);
	}) / (x.size() * (x.size() * x.size() - 1));
}

double Statistics::variance(Keys data) const noexcept
{
	return std::sqrt(std::transform_reduce(data.begin(), data.end(), 0.0, std::plus<>(), [aver = average(data)](double val) -> double {
		return (val - aver) * (val - aver);
	}) / data.size());
}

}
