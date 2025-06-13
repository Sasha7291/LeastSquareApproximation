#pragma once

#include "lsa_common.hpp"


namespace lsa
{

class Solver
{

public:
    Solver() = default;
    ~Solver() = default;

    Solver(const Solver &) = delete;
    Solver(Solver &&) = delete;
    Solver &operator=(const Solver &) = delete;
    Solver &operator=(Solver &&) = delete;

    template<NumberType T>
    [[nodiscard]] std::optional<UnknownColumn<T>> operator()(
        const CoefficientMatrix<T> &A,
        const FreeMemberColumn<T> &B
    ) const;

};

template<NumberType T>
std::optional<UnknownColumn<T>> Solver::operator()(
    const CoefficientMatrix<T> &A,
    const FreeMemberColumn<T> &B
) const
{
    try
    {
        return std::make_optional(A.inverted().toMatrix() * B);
    }
    catch (const dynamic_matrix::Exception &exception)
    {
#ifdef QT
        qDebug() << exception.what();
#else
        std::cout << exception.what();
#endif
        return std::nullopt;
    }
}
	
}
