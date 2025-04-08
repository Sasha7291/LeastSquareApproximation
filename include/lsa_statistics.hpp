#pragma once

#include <span>
#include <unordered_map>


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
	
	[[nodiscard]] double average(std::span<double> data) noexcept const;
	[[nodiscard]] double pearsonCoefficient(std::span<double> x, std::span<double> y) noexcept const;
	[[nodiscard]] std::vector<double> rank(std::span<double> data) noexcept const;
	[[nodiscard]] double spearmanCoefficient(std::span<double> x, std::span<double> y) noexcept const;
	[[nodiscard]] double variance(std::span<double> data) noexcept const;
	
};

double average(std::span<double> data) noexcept const
{
	return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double pearsonCoefficient(std::span<double> x, std::span<double> y) noexcept const
{
	return std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<>(), [xAverage = average(x), yAverage = average(y)](double val1, double val2) -> double {
		return (val1 - xAverage) * (val2 - yAverage);
	}) / (data.size() * variance(x) * variance(y));
}

std::vector<double> rank(std::span<double> data) noexcept const
{
	std::vector<double> rankData(data.size());
	auto min = *std::ranges::min_element(data);
	
	for (auto i = 0ull; i < data.size(); )
	{
		auto minIter = std::lower_bound(data.begin(), data.end(), min);
		min = *minIter;
		rankData[std::distance(minIter, data.begin())] = i;
	}
	
	return rankData;
}

double spearmanCoefficient(std::span<double> x, std::span<double> y) noexcept const
{
	return 1.0 - 6.0 * std::transform_reduce(rankX.cbegin(), rankX.cend(), rankY.cbegin(), 0.0, std::plus<>(), [rankX = rank(x), rankY = rank(y)](double rank1, double rank2) -> double {
		return (rank1 - rank2) * (rank1 - rank2);
	}) / (x.size() * (x.size() * x.size() - 1));
}

double variance(std::span<double> data) noexcept const
{
	return std::sqrt(std::transform_reduce(data.begin(), data.end(), 0.0, std::plus<>(), [aver = average(data)](double val) -> double {
		return (val - aver) * (val - aver);
	}) / data.size());
}

}