#pragma once
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <istream>
#include <map>
#include <queue>

#ifdef DEBUG
#    define debug(x)    std::cerr << #x << " = " << (x) << " "
#    define debugln(x)  std::cerr << #x << " = " << (x) << std::endl
#    define debugf(...) fprintf(stderr, __VA_ARGS__)
#    define debugv(fmt, ...)                                    \
        fprintf(stderr, "%s/%d: " fmt "\n", __func__, __LINE__, \
                __VA_OPT__(, ) __VA_ARGS__)
#else
#    define debug(x)
#    define debugln(x)
#    define debugf(...)
#    define debugv(...)
#endif

namespace autodiff {

enum class Func {
    none,
    oppo,   // -a
    add,    // a + b
    sub,    // a - b
    mul,    // a * b
    div,    // a / b
    log,    // log(a)
    exp,    // exp(a)
    sin,    // sin(a)
    cos,    // cos(a)
    tan,    // tan(a)
    asin,   // asin(a)
    acos,   // acos(a)
    atan,   // acos(a)
    sinh,   // sinh(a)
    cosh,   // cosh(a)
    tanh,   // tanh(a)
    sqrt,   // sqrt(a)
    power,  // a ** b
    abs,    // abs(a)
};

constexpr const char* func_name(Func id) {
    switch (id) {
        case Func::none: return "none";
        case Func::oppo: return "oppo";
        case Func::add: return "add";
        case Func::sub: return "sub";
        case Func::mul: return "mul";
        case Func::div: return "div";
        case Func::log: return "log";
        case Func::exp: return "exp";
        case Func::sin: return "sin";
        case Func::cos: return "cos";
        case Func::tan: return "tan";
        case Func::asin: return "asin";
        case Func::acos: return "acos";
        case Func::atan: return "atan";
        case Func::abs: return "abs";
        case Func::power: return "pow";
        case Func::sqrt: return "sqrt";
        case Func::sinh: return "sinh";
        case Func::cosh: return "cosh";
        case Func::tanh: return "tanh";
    }
    return "unknown";
}

template <typename T> class TapeNode {
public:
    Func func{Func::none};
    TapeNode* left{nullptr};
    TapeNode* right{nullptr};
    T value{0}, diff{0};
    int count{0};  // reference count
    bool require_diff{true};

    explicit TapeNode(T value, Func func = Func::none, TapeNode* left = nullptr,
                      TapeNode* right = nullptr)
        : func(func), left(left), right(right), value(value), diff(0), count(0) {
        if (left != nullptr) left->count++;
        if (right != nullptr) right->count++;
    }

    std::string id() const { return std::format("#{:02X}", ((size_t)this & 0xfff) >> 4); }

    std::string name() const {
        return std::format("#{:02X}:{:.4}:{}/{}", ((size_t)this & 0xfff) >> 4,
                           this->value, this->diff, this->count);
    }

    std::string to_string() const {
        return std::format("node(id: {}, func: {}, l/r: {}/{}, v: {}, d: {}, ref: {})",
                           this->id(), func_name(func), left ? left->id() : "   ",
                           right ? right->id() : "   ", value, diff, count);
    }

    friend std::ostream& operator<<(std::ostream& os, const TapeNode& v) {
        os << v.to_string();
        return os;
    }
    friend std::istream& operator>>(std::istream& is, TapeNode& v) {
        is >> v.value;
        return is;
    }

    void propagate() {
        std::map<TapeNode*, int> deg;
        std::function<void(TapeNode*)> find = [&](TapeNode* v) {
            if (deg.count(v)) return;
            deg[v] = 0;
            auto l = v->left, r = v->right;
            if (l != nullptr) find(l), deg[l]++;
            if (r != nullptr) find(r), deg[r]++;
        };
        find(this);
        std::queue<TapeNode*> q;
        this->diff = 1;
        q.push(this);
        while (q.size()) {
            TapeNode* cur = q.front();
            q.pop();
            TapeNode *l = cur->left, *r = cur->right;
            switch (cur->func) {
                case Func::none: continue;
                case Func::oppo: deg[l]--, l->diff -= cur->diff; break;
                case Func::add:
                    deg[l]--, l->diff += cur->diff;
                    deg[r]--, r->diff += cur->diff;
                    break;
                case Func::sub:
                    deg[l]--, l->diff += cur->diff;
                    deg[r]--, r->diff -= cur->diff;
                    break;
                case Func::mul:
                    deg[l]--, l->diff += cur->diff * r->value;
                    deg[r]--, r->diff += cur->diff * l->value;
                    break;
                case Func::div:
                    deg[l]--, l->diff += cur->diff / r->value;
                    deg[r]--, r->diff -= cur->diff * l->value / (r->value * r->value);
                    break;
                case Func::sqrt:
                    deg[l]--, l->diff += cur->diff / (2 * std::sqrt(l->value));
                    break;
                case Func::abs:
                    deg[l]--, l->diff += cur->diff * (l->value >= 0 ? 1 : -1);
                    break;
                case Func::log: deg[l]--, l->diff += cur->diff / l->value; break;
                case Func::exp: deg[l]--, l->diff += cur->diff * cur->value; break;
                case Func::sin:
                    deg[l]--, l->diff += cur->diff * std::cos(l->value);
                    break;
                case Func::cos:
                    deg[l]--, l->diff -= cur->diff * std::sin(l->value);
                    break;
                case Func::tan:
                    deg[l]--,
                        l->diff += cur->diff / (std::cos(l->value) * std::cos(l->value));
                    break;
                case Func::asin:
                    deg[l]--, l->diff += cur->diff / std::sqrt(1 - l->value * l->value);
                    break;
                case Func::acos:
                    deg[l]--, l->diff -= cur->diff / std::sqrt(1 - l->value * l->value);
                    break;
                case Func::atan:
                    deg[l]--, l->diff += cur->diff / (1 + l->value * l->value);
                    break;
                case Func::power:
                    deg[l]--, l->diff +=
                              cur->diff * r->value * std::pow(l->value, r->value - 1);
                    deg[r]--, r->diff += cur->diff * std::pow(l->value, r->value) *
                                         std::log(l->value);
                    break;
                case Func::sinh:
                    deg[l]--, l->diff += cur->diff * std::cosh(l->value);
                    break;
                case Func::cosh:
                    deg[l]--, l->diff += cur->diff * std::sinh(l->value);
                    break;
                case Func::tanh:
                    deg[l]--, l->diff +=
                              cur->diff / (std::cosh(l->value) * std::cosh(l->value));
                    break;
            }
            if (l != nullptr && !deg[l]) q.push(l);
            if (r != l && r != nullptr && !deg[r]) q.push(r);
        }
    }

    void remove() {
        if (left != nullptr) {
            left->count--;
            if (!left->count) {
                left->remove();
                left = nullptr;
                delete left;
            }
        }
        if (right != nullptr) {
            right->count--;
            if (!right->count) {
                right->remove();
                delete right;
                right = nullptr;
            }
        }
        left = right = nullptr;
        func = Func::none;
    }

    void print() {
        std::cerr << std::format("{}", to_string()) << std::endl;
        if (left != nullptr) {
            std::cerr << std::format("{0} ---{2}--> {1}", left->id(), id(),
                                     func_name(func))
                      << std::endl;
            left->print();
        }
        if (right != nullptr) {
            std::cerr << std::format("{0} ---{2}--> {1}", right->id(), id(),
                                     func_name(func))
                      << std::endl;
            right->print();
        }
    }
};

template <typename T> class Variable {
public:
    T value;
    TapeNode<T>* node;

    T raw() const { return value; }
    T diff() const { return node->diff; }
    T operator()() { return value; }

    Variable(T value = 0) : value(value), node(new TapeNode<T>(value)) { node->count++; }

    Variable(T value, Func func, TapeNode<T>* left, TapeNode<T>* right) : value(value) {
        node = new TapeNode<T>(value, func, left, right);
        node->count++;
    }

    Variable(const Variable& v) : value(v.value), node(v.node) { node->count++; }

    Variable(Variable&& v) noexcept : value(v.value), node(v.node) { v.node = nullptr; }

    ~Variable() {
        if (node != nullptr) {
            node->count--;
            if (node->count == 0) {
                node->remove();
                delete node;
                node = nullptr;
            }
        }
    }

    Variable& operator=(const Variable& other) {
        if (this == &other) return *this;
        Variable::~Variable();
        node = other.node;
        value = other.value;
        node->count++;
        return *this;
    }

    Variable& operator=(Variable&& other) noexcept {
        if (this == &other) return *this;
        Variable::~Variable();
        node = other.node;
        value = other.value;
        other.node = nullptr;
        return *this;
    }

    bool operator==(const Variable& other) const {
        return abs(value - other.value) < 1e-10;
    }
    auto operator<=>(const Variable& other) const = default;

    void clear() { this->node->diff = 0; }
    void propagate(bool remain = false) {
        if (node == nullptr) {
            std::cerr << "propagate nullptr" << std::endl;
            return;
        }
        node->propagate();
        if (!remain) {
            node->remove();
        }
    }
    void require_diff(bool require_diff) { node->require_diff = require_diff; }

    template <typename... Args> auto derivative(const Args&... args) {
        propagate();
        return std::make_tuple(args.diff()...);
    }

    friend std::ostream& operator<<(std::ostream& os, const Variable& v) {
        os << v.node->value;
        return os;
    }
    friend std::istream& operator>>(std::istream& os, const Variable& v) {
        os >> v.node->value;
        return os;
    }

    friend Variable operator+(const Variable& v) { return Variable(v); }
    friend Variable operator+(const Variable& a, const Variable& b) {
        return Variable(a.node->value + b.node->value, Func::add, a.node, b.node);
    }
    friend Variable operator-(const Variable& v) {
        return Variable(-v.node->value, Func::oppo, v.node);
    }
    friend Variable operator-(const Variable& a, const Variable& b) {
        return Variable(a.node->value - b.node->value, Func::sub, a.node, b.node);
    }
    friend Variable operator*(const Variable& a, const Variable& b) {
        return Variable(a.node->value * b.node->value, Func::mul, a.node, b.node);
    }
    friend Variable operator/(const Variable& a, const Variable& b) {
        return Variable(a.node->value / b.node->value, Func::div, a.node, b.node);
    }
    friend Variable operator^(const Variable& a, const Variable& b) { return pow(a, b); }
    friend Variable log(const Variable& v) {
        return Variable(std::log(v.node->value), Func::log, v.node, nullptr);
    }
    friend Variable sin(const Variable& v) {
        return Variable(std::sin(v.node->value), Func::sin, v.node, nullptr);
    }
    friend Variable cos(const Variable& v) {
        return Variable(std::cos(v.node->value), Func::cos, v.node, nullptr);
    }
    friend Variable tan(const Variable& v) {
        return Variable(std::tan(v.node->value), Func::tan, v.node, nullptr);
    }
    friend Variable exp(const Variable& v) {
        return Variable(std::exp(v.node->value), Func::exp, v.node, nullptr);
    }
    friend Variable sqrt(const Variable& v) {
        return Variable(std::sqrt(v.node->value), Func::sqrt, v.node, nullptr);
    }
    friend Variable asin(const Variable& v) {
        return Variable(std::asin(v.node->value), Func::asin, v.node, nullptr);
    }
    friend Variable acos(const Variable& v) {
        return Variable(std::acos(v.node->value), Func::acos, v.node, nullptr);
    }
    friend Variable atan(const Variable& v) {
        return Variable(std::atan(v.node->value), Func::atan, v.node, nullptr);
    }
    friend Variable pow(const Variable& a, const Variable& b) {
        return Variable(std::pow(a.node->value, b.node->value), Func::power, a.node,
                        b.node);
    }
    friend Variable sinh(const Variable& v) {
        return Variable(std::sinh(v.node->value), Func::sinh, v.node, nullptr);
    }
    friend Variable cosh(const Variable& v) {
        return Variable(std::cosh(v.node->value), Func::cosh, v.node, nullptr);
    }
    friend Variable tanh(const Variable& v) {
        return Variable(std::tanh(v.node->value), Func::tanh, v.node, nullptr);
    }
};
using var = Variable<double>;

};  // namespace autodiff

template <typename... Args> void clear(Args... v) { (v.clear(), ...); }

template <typename T> struct std::formatter<autodiff::Variable<T>> : std::formatter<T> {
    auto format(const autodiff::Variable<T>& v, std::format_context& ctx) const {
        return std::formatter<T>::format(v.value, ctx);
    }
};
