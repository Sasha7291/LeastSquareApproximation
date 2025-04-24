#pragma once

#include "lsa_solver.hpp"
#include "psr_average.h"
#include "psr_polynomial.h"
#include "psr_standartize.h"
#include "psr_variance.h"

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

    [[nodiscard]] Result linear(Keys x, Values y) const;
    [[nodiscard]] Result polynomial(Keys x, Values y, const std::size_t N) const;
	
private:
    [[nodiscard]] constexpr unsigned binomialCoefficient(unsigned n, unsigned k) const noexcept;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Type average, Type variance) const noexcept;
    [[nodiscard]] Coefficients coefficientReverseStandardization(Coefficients coeffs, Keys data) const noexcept;
    [[nodiscard]] constexpr unsigned factorial(unsigned n) const noexcept;

};

}
