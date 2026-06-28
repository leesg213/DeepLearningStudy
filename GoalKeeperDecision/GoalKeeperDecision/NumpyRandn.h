//
//  NumpyRandn.h
//  CatClassifierWithDeepNN
//
//  Created by Sugu Lee on 6/28/26.
//

#ifndef NumpyRandn_h
#define NumpyRandn_h

#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstddef>

class NumpyRandn
{
public:
    explicit NumpyRandn(std::uint32_t seed = 0)
        : generator_(seed)
    {
    }

    void seed(std::uint32_t seed)
    {
        generator_.seed(seed);
        has_gauss_ = false;
        cached_gauss_ = 0.0;
    }

    std::vector<double> randn(std::size_t n)
    {
        std::vector<double> result;
        result.reserve(n);

        for (std::size_t i = 0; i < n; ++i)
        {
            result.push_back(standard_normal());
        }

        return result;
    }
    std::vector<float> randnf(std::size_t n)
    {
        std::vector<float> result;
        result.reserve(n);

        for (std::size_t i = 0; i < n; ++i)
        {
            result.push_back(standard_normal());
        }

        return result;
    }
private:
    std::mt19937 generator_;

    bool has_gauss_ = false;
    double cached_gauss_ = 0.0;

    std::uint32_t next_uint32()
    {
        return generator_();
    }

    // Matches NumPy RandomState's conversion from MT19937 uint32s to double in [0, 1)
    double random_double()
    {
        std::uint32_t a = next_uint32() >> 5;
        std::uint32_t b = next_uint32() >> 6;

        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    // Matches NumPy legacy RandomState standard_normal behavior
    double standard_normal()
    {
        if (has_gauss_)
        {
            has_gauss_ = false;
            return cached_gauss_;
        }

        double x1;
        double x2;
        double r2;

        do
        {
            x1 = 2.0 * random_double() - 1.0;
            x2 = 2.0 * random_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);

        const double f = std::sqrt(-2.0 * std::log(r2) / r2);

        cached_gauss_ = f * x1;
        has_gauss_ = true;

        return f * x2;
    }
};

#endif /* NumpyRandn_h */
