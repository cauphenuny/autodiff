#pragma once
#include "lib/magic_enum.hpp"
#include "propagate.hpp"
#include "util.hpp"

#include <cmath>
#include <format>
#include <iostream>
#include <istream>

template <typename T> class Arithmetic : public Operation<T> {
public:
    // clang-format off
    enum class Type {
        none, oppo, add, sub, mul, div,
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
                case Type::none: return 1;
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
    std::tuple<T, T> backward(const T& diff, const T& lhs, const T& rhs) const override {
        using namespace std;
        auto [dl, dr] = [&]() -> std::tuple<T, T> {
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
        return {dl * diff, dr * diff};
    }
    Arithmetic(Type type) : type(type) {
        switch (type) {
            case Type::add:
            case Type::sub:
            case Type::mul:
            case Type::div:
            case Type::power: this->op_type = Operation<T>::OpType::binary; break;
            case Type::none: this->op_type = Operation<T>::OpType::none; break;
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
    TapeNode<T>* node;

    const T& raw() const override { return node->value(); }
    T& raw() override { return node->value(); }
    T diff() const override { return node->diff(); }
    T initial_diff() const override { return 1; }
    void clear() override { this->node->clear(); }

    Variable(T value = 0)
        : node(new TapeNode<T>(value, func_table<T>(Arithmetic<T>::Type::none))) {}

    Variable(T value, Arithmetic<T>::Type type, TapeNode<T>* left,
             TapeNode<T>* right = nullptr)
        : node(new TapeNode<T>(value, func_table<T>(type), left, right)) {}

    Variable(const Variable& v) : node(v.node) { node->add_ref(); }

    Variable(Variable&& v) noexcept : node(v.node) { v.node = nullptr; }

    ~Variable() {
        if (node != nullptr) {
            node->remove_ref();
            if (!node->ref_count()) delete node;
            node = nullptr;
        }
    }

    Variable& operator=(const Variable& other) {
        if (this == &other) return *this;
        Variable::~Variable();
        node = other.node;
        node->add_ref();
        return *this;
    }

    Variable& operator=(Variable&& other) noexcept {
        if (this == &other) return *this;
        Variable::~Variable();
        node = other.node;
        other.node = nullptr;
        return *this;
    }

    bool operator==(const Variable& other) const {
        return abs(raw() - other.raw()) < 1e-10;
    }
    auto operator<=>(const Variable& other) const { return raw() - other.raw(); }

    void propagate(bool remain_graph = false) override {
        if (node == nullptr) {
            runtimeError("propagate nullptr");
        }
        node->propagate();
        if (!remain_graph) {
            node->remove();
            node = nullptr;
        }
    }
    void require_diff(bool require_diff) { node->require_diff(require_diff); }

    template <typename... Args> auto derivative(const Args&... args) {
        propagate();
        return std::make_tuple(args.diff()...);
    }

    friend std::ostream& operator<<(std::ostream& os, const Variable& v) {
        os << v.raw();
        return os;
    }
    friend std::istream& operator>>(std::istream& os, Variable& v) {
        os >> v.raw();
        return os;
    }

    friend Variable operator+(const Variable& v) { return Variable(v); }
    friend Variable operator+(const Variable& a, const Variable& b) {
        return Variable(a.raw() + b.raw(), Arithmetic<T>::Type::add, a.node, b.node);
    }
    friend Variable operator-(const Variable& v) {
        return Variable(-v.raw(), Arithmetic<T>::Type::oppo, v.node);
    }
    friend Variable operator-(const Variable& a, const Variable& b) {
        return Variable(a.raw() - b.raw(), Arithmetic<T>::Type::sub, a.node, b.node);
    }
    friend Variable operator*(const Variable& a, const Variable& b) {
        return Variable(a.raw() * b.raw(), Arithmetic<T>::Type::mul, a.node, b.node);
    }
    friend Variable operator/(const Variable& a, const Variable& b) {
        return Variable(a.raw() / b.raw(), Arithmetic<T>::Type::div, a.node, b.node);
    }
    friend Variable operator^(const Variable& a, const Variable& b) { return pow(a, b); }
    friend Variable operator+=(const Variable& a, const Variable& b) { a = a + b; }
    friend Variable operator-=(const Variable& a, const Variable& b) { a = a - b; }
    friend Variable operator*=(const Variable& a, const Variable& b) { a = a * b; }
    friend Variable operator/=(const Variable& a, const Variable& b) { a = a / b; }
    friend Variable log(const Variable& v) {
        return Variable(std::log(v.raw()), Arithmetic<T>::Type::log, v.node);
    }
    friend Variable sin(const Variable& v) {
        return Variable(std::sin(v.raw()), Arithmetic<T>::Type::sin, v.node);
    }
    friend Variable cos(const Variable& v) {
        return Variable(std::cos(v.raw()), Arithmetic<T>::Type::cos, v.node);
    }
    friend Variable tan(const Variable& v) {
        return Variable(std::tan(v.raw()), Arithmetic<T>::Type::tan, v.node);
    }
    friend Variable exp(const Variable& v) {
        return Variable(std::exp(v.raw()), Arithmetic<T>::Type::exp, v.node);
    }
    friend Variable sqrt(const Variable& v) {
        return Variable(std::sqrt(v.raw()), Arithmetic<T>::Type::sqrt, v.node);
    }
    friend Variable asin(const Variable& v) {
        return Variable(std::asin(v.raw()), Arithmetic<T>::Type::asin, v.node);
    }
    friend Variable acos(const Variable& v) {
        return Variable(std::acos(v.raw()), Arithmetic<T>::Type::acos, v.node);
    }
    friend Variable atan(const Variable& v) {
        return Variable(std::atan(v.raw()), Arithmetic<T>::Type::atan, v.node);
    }
    friend Variable pow(const Variable& a, const Variable& b) {
        return Variable(std::pow(a.raw(), b.raw()), Arithmetic<T>::Type::power, a.node,
                        b.node);
    }
    friend Variable sinh(const Variable& v) {
        return Variable(std::sinh(v.raw()), Arithmetic<T>::Type::sinh, v.node);
    }
    friend Variable cosh(const Variable& v) {
        return Variable(std::cosh(v.raw()), Arithmetic<T>::Type::cosh, v.node);
    }
    friend Variable tanh(const Variable& v) {
        return Variable(std::tanh(v.raw()), Arithmetic<T>::Type::tanh, v.node);
    }
    friend Variable abs(const Variable& v) {
        return Variable(std::abs(v.raw()), Arithmetic<T>::Type::abs, v.node);
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