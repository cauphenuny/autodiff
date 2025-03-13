#pragma once
#include "autodiff.hpp"
#include "lib/magic_enum.hpp"
#include "util.hpp"

#include <cmath>
#include <format>

template <typename T> class Arithmetic : public Operation<T> {
public:
    // clang-format off
    enum class Type {
        oppo, add, sub, mul, div,
        log, exp, sin, cos, tan,
        asin, acos, atan, sinh, cosh, tanh,
        sqrt, power, abs
    };
    // clang-format on
    Type type;
    T backward(const T& diff, const T& arg) const override {
        using namespace std;
        auto coef = [&]() -> T {
            switch (type) {
                case Type::oppo: return -1;
                case Type::sqrt: return 0.5 / sqrt(arg);
                case Type::abs: return arg >= 0 ? 1 : -1;
                case Type::log: return 1 / arg;
                case Type::exp: return exp(arg);
                case Type::sin: return cos(arg);
                case Type::cos: return -sin(arg);
                case Type::tan: return 1 / (cos(arg) * cos(arg));
                case Type::asin: return 1 / sqrt(1 - arg * arg);
                case Type::acos: return -1 / sqrt(1 - arg * arg);
                case Type::atan: return 1 / (1 + arg * arg);
                case Type::sinh: return cosh(arg);
                case Type::cosh: return sinh(arg);
                case Type::tanh: return 1 / (cosh(arg) * cosh(arg));
                default:
                    runtimeError("invalid func type {} for unary backward",
                                 magic_enum::enum_name(type));
            }
        }();
        return coef * diff;
    }
    T forward(const T& arg) const override {
        using namespace std;
        switch (type) {
            case Type::oppo: return -arg;
            case Type::sqrt: return sqrt(arg);
            case Type::abs: return abs(arg);
            case Type::log: return log(arg);
            case Type::exp: return exp(arg);
            case Type::sin: return sin(arg);
            case Type::cos: return cos(arg);
            case Type::tan: return tan(arg);
            case Type::asin: return asin(arg);
            case Type::acos: return acos(arg);
            case Type::atan: return atan(arg);
            case Type::sinh: return sinh(arg);
            case Type::cosh: return cosh(arg);
            case Type::tanh: return tanh(arg);
            default:
                runtimeError("invalid func type {} for unary forward",
                             magic_enum::enum_name(type));
        }
    }
    std::tuple<T, T> backward(const T& diff, const T& lhs, const T& rhs) const override {
        using namespace std;
        auto [coef_l, coef_r] = [&]() -> std::tuple<T, T> {
            switch (type) {
                case Type::add: return {1, 1};
                case Type::sub: return {1, -1};
                case Type::mul: return {rhs, lhs};
                case Type::div: return {1 / rhs, -lhs / (rhs * rhs)};
                case Type::power:
                    return {rhs * pow(lhs, rhs - 1), pow(lhs, rhs) * log(lhs)};
                default:
                    runtimeError("invalid func type {} for binary backward",
                                 magic_enum::enum_name(type));
            }
        }();
        return {coef_l * diff, coef_r * diff};
    }
    T forward(const T& lhs, const T& rhs) const override {
        using namespace std;
        switch (type) {
            case Type::add: return lhs + rhs;
            case Type::sub: return lhs - rhs;
            case Type::mul: return lhs * rhs;
            case Type::div: return lhs / rhs;
            case Type::power: return pow(lhs, rhs);
            default:
                runtimeError("invalid func type {} for binary forward",
                             magic_enum::enum_name(type));
        }
    }
    Arithmetic(Type type) : type(type) {
        switch (type) {
            case Type::add:
            case Type::sub:
            case Type::mul:
            case Type::div:
            case Type::power: this->op_type = Operation<T>::OpType::binary; break;
            default: this->op_type = Operation<T>::OpType::unary; break;
        }
    }
    std::string_view name() const override { return magic_enum::enum_name(type); }
};

template <typename T> class ArithmeticFuncTable {
public:
    std::map<typename Arithmetic<T>::Type, Arithmetic<T>> content;
    ArithmeticFuncTable() {
        const auto types = magic_enum::enum_values<typename Arithmetic<T>::Type>();
        for (const auto& type : types) {
            content.emplace(type, Arithmetic<T>(type));
        }
    }
    Arithmetic<T>* operator()(Arithmetic<T>::Type type) { return &content.at(type); }
};

template <typename T> ArithmeticFuncTable<T> func_table;

template <typename T> class Variable : public AutoDiff<T> {
public:
    T initial_diff() const override { return 1; }

    Variable(T value = 0) : AutoDiff<T>(value) {}

    Variable(Arithmetic<T>::Type type, const auto&... args)
        : AutoDiff<T>(func_table<T>(type), args...) {}

    bool operator==(const Variable& other) const {
        return abs(this->raw() - other.raw()) < 1e-10;
    }
    auto operator<=>(const Variable& other) const { return this->raw() - other.raw(); }

    friend Variable operator+(const Variable& v) { return Variable(v); }
    friend Variable operator+(const Variable& a, const Variable& b) {
        return Variable(Arithmetic<T>::Type::add, a, b);
    }
    friend Variable operator-(const Variable& v) {
        return Variable(Arithmetic<T>::Type::oppo, v);
    }
    friend Variable operator-(const Variable& a, const Variable& b) {
        return Variable(Arithmetic<T>::Type::sub, a, b);
    }
    friend Variable operator*(const Variable& a, const Variable& b) {
        return Variable(Arithmetic<T>::Type::mul, a, b);
    }
    friend Variable operator/(const Variable& a, const Variable& b) {
        return Variable(Arithmetic<T>::Type::div, a, b);
    }
    friend Variable operator^(const Variable& a, const Variable& b) { return pow(a, b); }
    friend Variable operator+=(const Variable& a, const Variable& b) { a = a + b; }
    friend Variable operator-=(const Variable& a, const Variable& b) { a = a - b; }
    friend Variable operator*=(const Variable& a, const Variable& b) { a = a * b; }
    friend Variable operator/=(const Variable& a, const Variable& b) { a = a / b; }
    friend Variable log(const Variable& v) {
        return Variable(Arithmetic<T>::Type::log, v);
    }
    friend Variable sin(const Variable& v) {
        return Variable(Arithmetic<T>::Type::sin, v);
    }
    friend Variable cos(const Variable& v) {
        return Variable(Arithmetic<T>::Type::cos, v);
    }
    friend Variable tan(const Variable& v) {
        return Variable(Arithmetic<T>::Type::tan, v);
    }
    friend Variable exp(const Variable& v) {
        return Variable(Arithmetic<T>::Type::exp, v);
    }
    friend Variable sqrt(const Variable& v) {
        return Variable(Arithmetic<T>::Type::sqrt, v);
    }
    friend Variable asin(const Variable& v) {
        return Variable(Arithmetic<T>::Type::asin, v);
    }
    friend Variable acos(const Variable& v) {
        return Variable(Arithmetic<T>::Type::acos, v);
    }
    friend Variable atan(const Variable& v) {
        return Variable(Arithmetic<T>::Type::atan, v);
    }
    friend Variable pow(const Variable& a, const Variable& b) {
        return Variable(Arithmetic<T>::Type::power, a, b);
    }
    friend Variable sinh(const Variable& v) {
        return Variable(Arithmetic<T>::Type::sinh, v);
    }
    friend Variable cosh(const Variable& v) {
        return Variable(Arithmetic<T>::Type::cosh, v);
    }
    friend Variable tanh(const Variable& v) {
        return Variable(Arithmetic<T>::Type::tanh, v);
    }
    friend Variable abs(const Variable& v) {
        return Variable(Arithmetic<T>::Type::abs, v);
    }
};

template <typename T> Variable<T> max(const Variable<T>& a, const Variable<T>& b) {
    if (a > b) return a;
    return b;
}
template <typename T> Variable<T> min(const Variable<T>& a, const Variable<T>& b) {
    if (a < b) return a;
    return b;
}

using var = Variable<double>;

template <typename... Args> void clear(Args... v) { (v.clear(), ...); }

template <typename T> struct std::formatter<Variable<T>> : std::formatter<T> {
    auto format(const Variable<T>& v, std::format_context& ctx) const {
        return std::formatter<T>::format(v.raw(), ctx);
    }
};