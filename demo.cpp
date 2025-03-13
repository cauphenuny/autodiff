#include "src/variable.hpp"

#include <cassert>
#include <cmath>
#include <format>
#include <iostream>

using namespace std;

template <typename T> T f(T x, T y, T z) { return log(x * z) + x * y - sin(y) + cosh(z); }

template <typename T> T g(T x, T y, T z) { return z * pow(x, y); }

auto num_diff(auto f, auto eps, auto... args) {
    return (f((args + eps)...) - f(args...)) / eps;
}

int main() {
    double eps = 1e-7, eqeps = 1e-4;
    double x0 = 2, y0 = 5, z0 = 3;

    var x = x0, y = y0, z = z0;
    var u = f(x, y, z);
    auto [ux, uy, uz] = u.derivative(x, y, z);
    cout << format("u = {:.5}, ux = {:.5}, uy = {:.5}, uz = {:.5}\n", u, ux, uy, uz);
    assert(abs((ux + uy + uz) - num_diff(f<double>, eps, x0, y0, z0)) < eqeps);
    clear(x, y, z);

    var v = g(x, y, z);
    v.propagate();
    auto vx = x.diff(), vy = y.diff(), vz = z.diff();
    cout << format("v = {:.5}, vx = {:.5}, vy = {:.5}, vz = {:.5}\n", v, vx, vy, vz);
    assert(abs((vx + vy + vz) - num_diff(g<double>, eps, x0, y0, z0)) < eqeps);
    x.clear(), y.clear(), z.clear();

    return 0;
}
