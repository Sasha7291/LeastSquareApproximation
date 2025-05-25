#pragma once

#include "lsa_solver.hpp"


namespace lsa
{

class Approximator
{

public:
    Approximator() = default;
    ~Approximator() = default;

    Approximator(const Approximator &) = delete;
    Approximator(Approximator &&) = delete;
    Approximator &operator=(const Approximator &) = delete;
    Approximator &operator=(Approximator &&) = delete;

    [[nodiscard]] Result linear(Keys x, Values y) const;
    [[nodiscard]] Result linear(Keys x, Keys y, Values z) const;
    [[nodiscard]] Result polynomial(Keys x, Values y, std::size_t N) const;
    [[nodiscard]] Result polynomial(Keys x, Keys y, Values z, std::size_t N) const;

private:
    [[nodiscard]] constexpr std::size_t binomialCoefficient(std::size_t n, std::size_t k) const;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const;
    [[nodiscard]] Coefficients coefficientReverseStandardization(
        Coefficients coeffs,
        Type averageX,
        Type varianceX,
        Type averageY,
        Type varianceY,
        Type averageZ,
        Type varianceZ
    ) const;
    [[nodiscard]] constexpr double monomial(std::size_t n, double x, double y) const;
    [[nodiscard]] std::pair<std::size_t, std::size_t> degreeToIndicies(std::size_t n) const;

};

}
