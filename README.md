```cpp
#include "autodiff.h"

#include <format>
#include <functional>
#include <iostream>

using namespace std;

template <typename T> T f(T x, T y, T z) { return log(x * z) + x * y - sin(y) + cosh(z); }

template <typename T> T g(T x, T y, T z) { return z * pow(x, y); }

double eps = 1e-12;

template <typename Func, typename... Args> double numerical_diff(Func f, Args... args)
{
    return (f((args + eps)...) - f(args...)) / eps;
}

int main()
{
    using T = var::value_type;
    var x = 2, y = 5, z = 3;
    var u = f(x, y, z);
    auto [ux, uy, uz] = u.derivative(x, y, z);
    cout << format("u = {}, ux = {}, uy = {}, uz = {}\n", u(), ux, uy, uz);
    assert(((ux + uy + uz) - numerical_diff(f<T>, x(), y(), z())) < eps);
    clear_diff(x, y, z);

    var v = g(x, y, z);
    v.propagate();
    cout << format("v = {}, vx = {}, vy = {}, vz = {}\n", v(), x.diff(), y.diff(), z.diff());
    assert(((x.diff() + y.diff() + z.diff()) - numerical_diff(g<T>, x(), y(), z())) < eps);
    x.clear(), y.clear(), z.clear();
    return 0;
}
```