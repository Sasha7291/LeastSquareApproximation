#include "lsa_approximator.h"


namespace lsa
{

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

Result Approximator::plane(Keys x, Keys y, Values z) const
{
    auto averX = psr::Average<double>{}(x);
    auto averY = psr::Average<double>{}(y);
    auto averZ = psr::Average<double>{}(z);
    auto varX = psr::Variance<double>{averX}(x);
    auto varY = psr::Variance<double>{averY}(y);
    auto varZ = psr::Variance<double>{averZ}(z);
    auto tempX = psr::Standartize<double>{averX, varX}(x);
    auto tempY = psr::Standartize<double>{averY, varY}(y);
    auto tempZ = psr::Standartize<double>{averZ, varZ}(z);

    dynamic_matrix::SquareMatrix<Type> A(2);
    dynamic_matrix::Matrix<Type> B(2, 1);

    std::function<double(Keys, Keys)> crossedSum = [](Keys keys1, Keys keys2) -> double {
        return std::transform_reduce(keys1.cbegin(), keys1.cend(), keys2.cbegin(), static_cast<Type>(0), std::plus<>(), [](Type value1, Type value2) -> Type {
            return value1 * value2;
        });
    };

    A(0, 0) = crossedSum(tempX, tempX);
    A(0, 1) = A(1, 0) = crossedSum(tempX, tempY);
    A(1, 1) = crossedSum(tempY, tempY);
    B(0, 0) = crossedSum(tempX, tempZ);
    B(0, 0) = crossedSum(tempY, tempZ);

    auto unknownColumn = Solver()(A, B);
    if (!unknownColumn.has_value())
        throw Exception("Coefficients matrix is irreversible");

    return {
        averZ - unknownColumn.value()(0, 0) * varZ * averX / varX - unknownColumn.value()(1, 0) * varZ * averY / varY,
        unknownColumn.value()(0, 0) * varZ / varX,
        unknownColumn.value()(1, 0) * varZ / varY
    };
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
        sums[i] = std::accumulate(tempX.cbegin(), tempX.cend(), static_cast<Type>(0), [i](Type total, Type value) -> Type {
            return total + std::pow(value, i);
        });

    for (auto i = 0ull; i < N; ++i)
    {
        for (auto j = i; j < N; ++j)
            A(i, j) = A(j, i) = sums[i + j];

        B(i, 0) = std::transform_reduce(tempX.cbegin(), tempX.cend(), y.cbegin(), static_cast<Type>(0), std::plus<>(), [i](Type value1, Type value2) -> Type {
            return std::pow(value1, i) * value2;
        });
    }

    auto unknownColumn = Solver()(A, B);
    if (!unknownColumn.has_value())
        throw Exception("Coefficients matrix is irreversible");

    return coefficientReverseStandardization(unknownColumn.value().column(0), aver, var);
}

constexpr unsigned Approximator::binomialCoefficient(unsigned n, unsigned k) const
{
    return factorial(n) / (factorial(k) * factorial(n - k));
}

constexpr unsigned Approximator::factorial(unsigned n) const
{
    return (n > 1) ? n * factorial(n - 1) : 1;
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

Coefficients Approximator::coefficientReverseStandardization(Coefficients coeffs, Keys data) const
{
    auto aver = psr::Average<double>{}(data);
    return coefficientReverseStandardization(coeffs, aver, psr::Variance<double>{aver}(data));
}

}
