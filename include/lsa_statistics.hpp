#pragma once

#include "lsa_arithmetic.hpp"

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

    [[nodiscard]] ResultValues operator()(Keys x, Values y, ResultValues yApp) const noexcept;

    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const noexcept;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept;
    [[nodiscard]] Type curveMaxDeflection(Values y1, Values y2) const noexcept;
    [[nodiscard]] Type curveVariance(Keys x, Values y1, Values y2) const noexcept;
    [[nodiscard]] Type pearsonCoefficient(Keys x, Values y, Type xAver, Type yAver, Type xVar, Type yVar) const noexcept;
    [[nodiscard]] Type pearsonCoefficient(Keys x, Values y) const noexcept;
    [[nodiscard]] Type spearmanCoefficient(Keys x, Values y) const noexcept;
	
};

ResultValues Statistics::operator ()(Keys x, Values y, ResultValues yApp) const noexcept
{
    return {
        pearsonCoefficient(x, y),
        spearmanCoefficient(x, y),
        curveVariance(x, y, yApp),
        curveMaxDeflection(y, yApp)
    };
}

Coefficients Statistics::coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const noexcept
{
    Arithmetic arithmetic;
    const auto n = coeffs.size();
    Coefficients result(n, 0);

    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i; j < n; ++j)
            result[i] += coeffs[j] * arithmetic.binomialCoefficient(j, i) * std::pow(-average, j - i) / std::pow(variance, j);

    return result;
}

Coefficients Statistics::coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept
{
    Arithmetic arithmetic;
    auto aver = arithmetic.average(data);
    return coefficientReverseStandardization(coeffs, aver, arithmetic.variance(data, aver));
}

Type Statistics::curveMaxDeflection(Values y1, Values y2) const noexcept
{
    ResultValues diff(y1.size());

    std::transform(y1.cbegin(), y1.cend(), y2.cbegin(), diff.begin(), [](Type val1, Type val2) -> Type {
        return std::abs(val1 - val2);
    });

    return *std::ranges::max_element(diff);
}

Type Statistics::curveVariance(Keys x, Values y1, Values y2) const noexcept
{
    ResultValues diff(y1.size());

    std::transform(y1.cbegin(), y1.cend(), y2.cbegin(), diff.begin(), [](Type val1, Type val2) -> Type {
        return (val1 - val2) * (val1 - val2);
    });

    return std::sqrt(Arithmetic{}.integrate(x, diff) / (x.back() - x.front()));
}

Type Statistics::pearsonCoefficient(Keys x, Values y, Type xAver, Type yAver, Type xVar, Type yVar) const noexcept
{
    return std::transform_reduce(x.cbegin(), x.cend(), y.cbegin(), 0.0, std::plus<>(), [xAver, yAver](Type val1, Type val2) -> Type {
        return (val1 - xAver) * (val2 - yAver);
    }) / (x.size() * xVar * yVar);
}

Type Statistics::pearsonCoefficient(Keys x, Values y) const noexcept
{
    Arithmetic arithmetic;
    auto xAver = arithmetic.average(x);
    auto yAver = arithmetic.average(y);
    return pearsonCoefficient(x, y, xAver, yAver, arithmetic.variance(x, xAver), arithmetic.variance(y, yAver));
}

Type Statistics::spearmanCoefficient(Keys x, Values y) const noexcept
{
    Arithmetic arithmetic;
    auto rankX = arithmetic.rank(x);
    auto rankY = arithmetic.rank(y);

    return 1.0 - 6.0 * std::transform_reduce(rankX.cbegin(), rankX.cend(), rankY.cbegin(), 0.0, std::plus<>(), [](Type rank1, Type rank2) -> Type {
		return (rank1 - rank2) * (rank1 - rank2);
	}) / (x.size() * (x.size() * x.size() - 1));
}

}
