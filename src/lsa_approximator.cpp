#include "lsa_approximator.h"

#include "psr_average.h"
#include "psr_standartize.h"
#include "psr_variance.h"

#include <numeric>


namespace lsa
{

Result Approximator::linear(Keys x, Values y) const
{
    try
    {
        return polynomial(x, y, 2ull);
    }
    catch (const Exception &)
    {
        throw;
    }
}

Result Approximator::linear(Keys x, Keys y, Values z) const
{
    try
    {
        return polynomial(x, y, z, 2ull);
    }
    catch (const Exception &)
    {
        throw;
    }
}

Result Approximator::polynomial(Keys x, Values y, std::size_t N) const
{
    auto aver = psr::Average<double>{}(x);
    auto var = psr::Variance<double>{aver}(x);
    auto tempX = psr::Standartize<double>{aver, var}(x);

    dynamic_matrix::SquareMatrix<Type> A{N};
    dynamic_matrix::Matrix<Type> B{N, 1};

    ResultValues sums(2 * N - 1);
    for (qsizetype i = 0ll; i < sums.size(); ++i)
        sums[i] = std::accumulate(tempX.cbegin(), tempX.cend(), static_cast<Type>(0), [i](Type total, Type value) -> Type {
            return total + std::pow(value, i);
        });

    for (std::size_t i = 0ull; i < N; ++i)
    {
        for (std::size_t j = i; j < N; ++j)
            A(i, j) = A(j, i) = sums[i + j];

        B(i, 0) = std::transform_reduce(tempX.cbegin(), tempX.cend(), y.cbegin(), static_cast<Type>(0), std::plus<>(), [i](Type value1, Type value2) -> Type {
            return std::pow(value1, i) * value2;
        });
    }

    auto unknownColumn = Solver{}(A, B);
    if (!unknownColumn.has_value())
        throw Exception("Coefficients matrix is irreversible");

    return coefficientReverseStandardization(unknownColumn.value().column(0), aver, var);
}

Result Approximator::polynomial(Keys x, Keys y, Values z, std::size_t N) const
{
    auto averX = psr::Average<double>{}(x);
    auto varX = psr::Variance<double>{averX}(x);
    auto tempX = psr::Standartize<double>{averX, varX}(x);
    auto averY = psr::Average<double>{}(y);
    auto varY = psr::Variance<double>{averY}(y);
    auto tempY = psr::Standartize<double>{averY, varY}(y);
    auto averZ = psr::Average<double>{}(z);
    auto varZ = psr::Variance<double>{averZ}(z);
    auto tempZ = psr::Standartize<double>{averZ, varZ}(z);

    N = N * (N + 1) / 2;
    dynamic_matrix::SquareMatrix<double> A{N};
    dynamic_matrix::Matrix<double> B{N, 1};

    for (std::size_t i = 0ull; i < N; ++i)
    {
        for (std::size_t j = 0ull; j < N; ++j)
            A(i, j) = A(j, i) = std::transform_reduce(tempX.cbegin(), tempX.cend(), tempY.cbegin(), 0.0, std::plus<>(), [this, i, j](double value1, double value2) -> double {
                return monomial(i, value1, value2) * monomial(j, value1, value2);
            });

        double sum = 0.0;
        for (std::size_t k = 0ull; k < N; ++k)
            sum += tempZ[k] * monomial(i, tempX[k], tempY[k]);
    }

    auto unknownColumn = Solver{}(A, B);
    if (!unknownColumn.has_value())
        throw Exception("Coefficients matrix is irreversible");

    return coefficientReverseStandardization(unknownColumn.value().column(0), averX, varX, averY, varY, averZ, varZ);
}

constexpr std::size_t Approximator::binomialCoefficient(std::size_t n, std::size_t k) const
{
    return std::tgamma(n + 1) / (std::tgamma(k + 1) * std::tgamma(n - k + 1));
}

Coefficients Approximator::coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const
{
    const auto n = coeffs.size();
    Coefficients result(n, 0);

    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i; j < n; ++j)
            result[i] += coeffs[j] * binomialCoefficient(j, i) * std::pow(-average, j - i) / std::pow(variance, j);

    return result;
}

Coefficients Approximator::coefficientReverseStandardization(
    Coefficients coeffs,
    Type averageX,
    Type varianceX,
    Type averageY,
    Type varianceY,
    Type averageZ,
    Type varianceZ
) const
{
    Coefficients result(coeffs.size(), 0.0);

    for (std::size_t k = 0; k < coeffs.size(); ++k)
    {
        const auto [p, q] = degreeToIndicies(k);
        double factor = varianceZ / (std::pow(varianceX, p) * std::pow(varianceY, q));

        for (std::size_t i = 0; i <= p; ++i)
            for (std::size_t j = 0; j <= q; ++j)
                result[(i + j) * (i + j + 1) / 2 + i] +=
                    coeffs[k] * factor
                    * binomialCoefficient(p, i) * binomialCoefficient(q, j)
                    * pow(-averageX, p - i) * pow(-averageY, q - j);
    }

    result[0] += averageZ;
    return result;
}

constexpr double Approximator::monomial(std::size_t n, double x, double y) const
{
    const auto [p, q] = degreeToIndicies(n);
    return std::pow(x, p) * std::pow(y, q);
}

std::pair<std::size_t, std::size_t> Approximator::degreeToIndicies(std::size_t n) const
{
    std::size_t m = 0ull;
    while ((m + 1) * (m + 2) / 2 <= n)
        ++m;

    std::size_t p = n - m * (m + 1) / 2;
    std::size_t q = m - p;

    return std::make_pair(p, q);
}

}
