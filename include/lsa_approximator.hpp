#pragma once

#include "lsa_solver.hpp"

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

    [[nodiscard]] Result exponential(Keys x, Values y) const noexcept;
    [[nodiscard]] Result linear(Keys x, Values y) const noexcept;
    [[nodiscard]] Result polynomial(Keys x, Values y, const std::size_t N) const noexcept;
	
private:
    [[nodiscard]] ResultValues approximateValues(const Coefficients &coeffs, Keys x) const noexcept;

};

Result Approximator::exponential(Keys x, Values y) const noexcept
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

Result Approximator::linear(Keys x, Values y) const noexcept
{
    return polynomial(x, y, 2ull);
}

Result Approximator::polynomial(Keys x, Values y, const std::size_t N) const noexcept
{
    dynamic_matrix::SquareMatrix<double> A(N);
    dynamic_matrix::Matrix<double> B(N, 1);

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

    auto result = Solver()(A, B).column(0);
    return std::make_pair(result, approximateValues(result, x));
}

[[nodiscard]] ResultValues Approximator::approximateValues(const Coefficients &coeffs, Keys x) const noexcept
{
    ResultValues r;
    r.reserve(x.size());

    for (auto i = 0ull; i < x.size(); ++i)
    {
        double sum = 0.0;

        for (auto j = 0; j < coeffs.size(); ++j)
            sum += std::pow(x[i], j) * coeffs[j];

        r.push_back(sum);
    }
	
	return r;
}

}
