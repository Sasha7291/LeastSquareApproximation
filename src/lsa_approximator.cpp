#include "lsa_approximator.h"

#include "lsa_solver.hpp"

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
    auto aver = psr::Average<Type>{}(x);
    auto var = psr::Variance<Type>{aver}(x);
    auto tempX = psr::Standartize<Type>{aver, var}(x);

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
        throw Exception{"Coefficients matrix is irreversible"};

    return coefficientReverseStandardization(unknownColumn.value().column(0), aver, var);
}

Result Approximator::polynomial(Keys x, Keys y, Values z, std::size_t degree) const
{
    const auto n = x.size();

    if (n < 3)
        throw Exception{"At least 3 points required"};

    const Type mx = psr::Average<Type>{}(x);
    const Type my = psr::Average<Type>{}(y);
    const Type sx = std::sqrt(psr::Variance<Type>{mx}(x));
    const Type sy = std::sqrt(psr::Variance<Type>{my}(y));

    if (sx < 1e-9 || sy < 1e-9)
        throw Exception{"Variance too small"};

    Type S[3][4] = {{0}};

    S[0][0] = n;

    for (std::size_t i = 0; i < n; ++i)
    {
        const Type ys = (y[i] - my) / sy;
        const Type xs = (x[i] - mx) / sx;

        S[0][1] += ys;
        S[0][2] += xs;
        S[0][3] += z[i];
        S[1][1] += ys * ys;
        S[1][2] += ys * xs;
        S[1][3] += ys * z[i];
        S[2][1] += xs * ys;
        S[2][2] += xs * xs;
        S[2][3] += xs * z[i];
    }

    S[1][0] = S[0][1];
    S[2][0] = S[0][2];
    S[2][1] = S[1][2];

    for (std::size_t col = 0; col < 3; ++col)
    {
        std::size_t pivot = col;

        for (std::size_t row = col + 1; row < 3; ++row)
            if (std::abs(S[row][col]) > std::abs(S[pivot][col]))
                pivot = row;

        std::swap(S[col], S[pivot]);

        if (std::abs(S[col][col]) < 1e-12)
            throw Exception{"Singular matrix"};

        const Type div = S[col][col];

        for (std::size_t j = col; j < 4; ++j)
            S[col][j] /= div;

        for (std::size_t row = 0; row < 3; ++row)
        {
            if (row == col)
                continue;

            const Type f = S[row][col];

            for (std::size_t j = col; j < 4; ++j)
                S[row][j] -= f * S[col][j];
        }
    }

    return {
        S[0][3] - S[1][3] * (my / sy) - S[2][3] * (mx / sx),
        S[1][3] / sy,
        S[2][3] / sx
    };
}

constexpr std::size_t Approximator::binomialCoefficient(std::size_t n, std::size_t k) const
{
    return std::tgamma(n + 1) / (std::tgamma(k + 1) * std::tgamma(n - k + 1));
}

Coefficients Approximator::coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const
{
    const auto n = coeffs.size();
    const auto sigma = std::sqrt(variance);
    Coefficients result(n, 0);

    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i; j < n; ++j)
            result[i] += coeffs[j] * binomialCoefficient(j, i) * std::pow(-average, j - i) / std::pow(sigma, j);

    return result;
}

Coefficients Approximator::coefficientReverseStandardization(
    Coefficients coeffs,
    Type averageX,
    Type varianceX,
    Type averageY,
    Type varianceY
) const
{
    Coefficients result(coeffs.size(), 0.0);
    const auto sigmaX = std::sqrt(varianceX);
    const auto sigmaY = std::sqrt(varianceY);

    for (decltype(coeffs.size()) k = 0; k < coeffs.size(); ++k)
    {
        const auto [p, q] = degreeToIndicies(k);

        double factor = 1.0 / (std::pow(sigmaX, p) * std::pow(sigmaY, q));

        for (std::size_t i = 0; i <= p; ++i)
            for (std::size_t j = 0; j <= q; ++j)
                result[(i + j) * (i + j + 1) / 2 + i] +=
                    coeffs[k] * factor
                    * binomialCoefficient(p, i) * binomialCoefficient(q, j)
                    * pow(-averageX, p - i) * pow(-averageY, q - j);
    }

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
