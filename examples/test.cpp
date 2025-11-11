#include "optim.hpp"
#include "tensor.hpp"

#include <random>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "lib/doctest.h"
#include "variable.hpp"

#include <cmath>

const double eps = 1e-7;

bool almost_equal(double a, double b) {
    std::stringstream ss_a, ss_b;
    ss_a << std::setprecision(6) << a;
    ss_b << std::setprecision(6) << b;
    return (ss_a.str() == ss_b.str()) || (std::abs(a - b) < 1e-6);
}

auto num_diff(auto f, auto eps, auto... args) {
    return (f((args + eps)...) - f((args - eps)...)) / eps / 2;
}

std::tuple<double, double> get_diff(auto f, auto... args) {
    auto var_args = std::make_tuple(var(args)...);
    auto u = std::apply(f, var_args);
    u.propagate();
    double expected =
        std::apply([](const auto&... args) { return (args.diff() + ...); }, var_args);
    double result = num_diff(f, eps, args...);
    return {expected, result};
}

bool check_diff(auto... args) {
    auto [expected, result] = get_diff(args...);
    return almost_equal(expected, result);
}

void check_impl(auto f, const std::string& name, auto... args) {
    auto [expected, result] = get_diff(f, args...);
    std::string args_str = ((std::to_string(args) + ", ") + ...);
    args_str.pop_back(), args_str.pop_back();
    CHECK_MESSAGE(almost_equal(expected, result), "in `", name, "`: autodiff ", expected,
                  " real ", result, " when args = [", args_str, "]");
}

#define check_func(func, ...) check_impl(func, #func, __VA_ARGS__)

TEST_CASE("add") {
    auto add = [](auto x, auto y) { return x + y; };
    check_func(add, -4, 3);
}

TEST_CASE("complex arithmetic operations") {
    auto func = [](auto x, auto y, auto z) { return x * y + x / z - y * z; };
    check_func(func, 2, 3, 4);
    check_func(func, 2, 5, 3);
    check_func(func, 100, 200, 500);
    check_func(func, -15, -20, -15);
}

TEST_CASE("log/exp/sin/cos/tan/abs") {
    auto func = [](auto x, auto y, auto z) {
        return log(abs(x * z + 1)) + exp(x) * sin(y) - x * cos(y) + tan(z) + sin(x * y) -
               exp(z) / (cos(x) + 1);
    };
    check_func(func, 2, 5, 3);
    check_func(func, 4, 10, 2);
    check_func(func, 2, -5, 3);
    check_func(func, -4, 10, 2);
}

TEST_CASE("asin/acos/atan") {
    auto func = [](auto x, auto y, auto z) {
        return asin(x) + acos(y) - atan(z) + asin(z) * acos(x);
    };
    check_func(func, 0.5, 0.3, 0.7);
    check_func(func, 0.1, 0.2, 0.3);
    check_func(func, -0.1, -0.2, -0.3);
}

TEST_CASE("sinh/cosh/tanh") {
    auto func = [](auto x, auto y, auto z) {
        return sinh(x) + cosh(y) + tanh(z) * sinh(x);
    };
    check_func(func, 1, 10, -20);
    check_func(func, -5, 14, -2);
    check_func(func, 10, 1, 10);
}

TEST_CASE("sqrt/power") {
    auto func = [](auto x, auto y, auto z) { return sqrt(x) + pow(y, 2) + pow(z, 3); };
    check_func(func, 4, 2, 3);
    check_func(func, 9, 3, 2);
    check_func(func, 16, 4, 1);
    check_func(func, 25, 5, 0);
}

TEST_CASE("compare") {
    var nan_number = std::nan("");
    var a = 1, b = 1, c = 2;
    CHECK(a == b);
    CHECK(nan_number != a);
    CHECK(a < c);
    CHECK(a <= c);
}

TEST_CASE("copy") {
    var a = 1, b = 2;
    var c = a + b;
    CHECK(a != c);
    CHECK(a != b);
    CHECK(almost_equal(c.raw(), 3));
    c = c + a;
    CHECK(almost_equal(c.raw(), 4));
}

TEST_CASE("fitting") {
    const int N = 100;
    std::vector<double> xs, ys;
    std::default_random_engine eng(42);
    std::normal_distribution<double> noise(0.0, 1.0);

    double k0 = 10.0, b0 = -5;
    for (int i = 0; i < N; i++) {
        double x = -10.0 + 20.0 * i / (N - 1);
        double y = 10.0 * x + b0 + noise(eng);
        xs.push_back(x);
        ys.push_back(y);
    }

    var k = 0, b = 0, loss = 0;
    // optim::GradientDescent optimizer({&k, &b}, 1e-3);
    optim::Adam optimizer{{&k, &b}, 1};
    const int iterations = 50;

    for (int iter = 0; iter < iterations; iter++) {
        var total_loss = 0.0;

        for (int i = 0; i < N; i++) {
            var diff = (k * xs[i] + b) - ys[i];
            total_loss = total_loss + diff * diff;
        }
        total_loss = total_loss / N;
        loss = total_loss;

        total_loss.propagate();
        optimizer.step();
    }

    INFO(std::format("fit result: (k0, b0) = ({}, {}), (k, b) = ({}, {}), loss = {}", k0,
                     b0, k, b, loss));
    CHECK(std::abs((k - k0).raw()) < 1.0);
    CHECK(std::abs((b - b0).raw()) < 1.0);
}

TEST_CASE("tensor") {
    // auto t = Tensor<double>::ones({2, 3, 4});
    // t[0, {0, 2}, {1, 3}] = Tensor<double>::zeros({2, 2});
    // CHECK(t[0, 0, 1] == 0);
}
