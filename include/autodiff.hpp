#pragma once
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <istream>
#include <map>
#include <queue>

enum ops {
    none,
    oppo,     // -a
    add,      // a + b
    sub,      // a - b
    mul,      // a * b
    divis,    // a / b
    ln,       // log(a)
    expo,     // exp(a)
    sine,     // sin(a)
    cosine,   // cos(a)
    tangent,  // tan(a)
    arcsin,   // asin(a)
    arccos,   // acos(a)
    arctan,   // acos(a)
    sin_h,    // sinh(a)
    cos_h,    // cosh(a)
    tan_h,    // tanh(a)
    sqroot,   // sqrt(a)
    power,    // a ^ b
    abso,     // abs(a)
};

constexpr const char* op_name(ops id)
{
    switch (id) {
        case ops::none: return "none";
        case ops::oppo: return "oppo";
        case ops::add: return "add";
        case ops::sub: return "sub";
        case ops::mul: return "mul";
        case ops::divis: return "div";
        case ops::ln: return "log";
        case ops::expo: return "exp";
        case ops::sine: return "sin";
        case ops::cosine: return "cos";
        case ops::tangent: return "tan";
        case ops::arcsin: return "asin";
        case ops::arccos: return "acos";
        case ops::arctan: return "atan";
        case ops::abso: return "abs";
        case ops::power: return "pow";
        case ops::sqroot: return "sqrt";
        case ops::sin_h: return "sinh";
        case ops::cos_h: return "cosh";
        case ops::tan_h: return "tanh";
    }
    return "unknown";
}

template <typename T> class TapeNode
{
public:
    ops op{none};
    TapeNode* left{nullptr};
    TapeNode* right{nullptr};
    T value{0}, diff{0};
    int count{0};  // reference count
    bool require_diff{true};

    TapeNode(T value, ops op = none, TapeNode* left = nullptr,
             TapeNode* right = nullptr)
        : value(value), op(op), left(left), right(right), diff(0), count(0)
    {
        if (left != nullptr) left->count++;
        if (right != nullptr) right->count++;
    }

    const std::string id() const
    {
        return std::format("#{:02X}", (((size_t)this) & 0xfff) >> 4);
    }

    const std::string name() const
    {
        return std::format("#{:02X}:{:.4}:{}/{}", (((size_t)this) & 0xfff) >> 4,
                           this->value, this->diff, this->count);
    }

    const std::string to_string() const
    {
        return std::format(
            "node(id: {}, op: {}, l/r: {}/{}, v: {}, d: {}, ref: {})",
            this->id(), op_name(op), left ? left->id() : "   ",
            right ? right->id() : "   ", value, diff, count);
    }

    friend std::ostream& operator<<(std::ostream& os, const TapeNode& v)
    {
        os << v.to_string();
        return os;
    }
    friend std::istream& operator>>(std::istream& is, TapeNode& v)
    {
        is >> v.value;
        return is;
    }

    void propagate()
    {
        std::map<TapeNode*, int> deg;
        std::function<void(TapeNode*)> find = [&](TapeNode* v) {
            if (deg.count(v)) return;
            deg[v] = 0;
            auto l = v->left, r = v->right;
            if (l != nullptr) find(l), deg[l]++;
            if (r != nullptr) find(r), deg[r]++;
        };
        find(this);
        // for (auto [p, cnt] : deg) {
        //     std::cerr << std::format("selected: {}, refcnt: {}", p->id(),
        //     cnt)
        //               << std::endl;
        // }
        std::queue<TapeNode*> q;
        this->diff = 1;
        q.push(this);
        while (q.size()) {
            TapeNode* cur = q.front();
            // debugln(cur->to_string());
            q.pop();
            // if (!cur->require_diff) continue;
            TapeNode *l = cur->left, *r = cur->right;
            switch (cur->op) {
                case ops::none: continue;
                case ops::oppo: deg[l]--, l->diff -= cur->diff; break;
                case ops::add:
                    deg[l]--, deg[r]--;
                    l->diff += cur->diff;
                    r->diff += cur->diff;
                    break;
                case ops::sub:
                    deg[l]--, deg[r]--;
                    l->diff += cur->diff;
                    r->diff -= cur->diff;
                    break;
                case ops::mul:
                    deg[l]--, deg[r]--;
                    l->diff += cur->diff * r->value;
                    r->diff += cur->diff * l->value;
                    break;
                case ops::divis:
                    deg[l]--, deg[r]--;
                    l->diff += cur->diff / r->value;
                    r->diff -= cur->diff * l->value / (r->value * r->value);
                    break;
                case ops::sqroot:
                    deg[l]--;
                    l->diff += cur->diff / (2 * std::sqrt(l->value));
                    break;
                case ops::abso:
                    deg[l]--;
                    l->diff += cur->diff * (l->value >= 0 ? 1 : -1);
                    break;
                case ops::ln:
                    deg[l]--;
                    l->diff += cur->diff / l->value;
                    break;
                case ops::expo:
                    deg[l]--;
                    l->diff += cur->diff * cur->value;
                    break;
                case ops::sine:
                    deg[l]--;
                    l->diff += cur->diff * std::cos(l->value);
                    break;
                case ops::cosine:
                    deg[l]--;
                    l->diff -= cur->diff * std::sin(l->value);
                    break;
                case ops::tangent:
                    deg[l]--;
                    l->diff +=
                        cur->diff / (std::cos(l->value) * std::cos(l->value));
                    break;
                case ops::arcsin:
                    deg[l]--;
                    l->diff += cur->diff / std::sqrt(1 - l->value * l->value);
                    break;
                case ops::arccos:
                    deg[l]--;
                    l->diff -= cur->diff / std::sqrt(1 - l->value * l->value);
                    break;
                case ops::arctan:
                    deg[l]--;
                    l->diff += cur->diff / (1 + l->value * l->value);
                    break;
                case ops::power:
                    deg[l]--, deg[r]--;
                    l->diff +=
                        cur->diff * r->value * std::pow(l->value, r->value - 1);
                    r->diff += cur->diff * std::pow(l->value, r->value) *
                               std::log(l->value);
                    break;
                case ops::sin_h:
                    deg[l]--;
                    l->diff += cur->diff * std::cosh(l->value);
                    break;
                case ops::cos_h:
                    deg[l]--;
                    l->diff += cur->diff * std::sinh(l->value);
                    break;
                case ops::tan_h:
                    deg[l]--;
                    l->diff +=
                        cur->diff / (std::cosh(l->value) * std::cosh(l->value));
                    break;
            }
            // std::cerr << std::format("left: {}, refcnt: {}\n", l->id(),
            // deg[l]); std::cerr << std::format("right: {}, refcnt: {}\n",
            // r->id(), deg[r]);
            if (l != nullptr && !deg[l]) q.push(l);
            if (r != l && r != nullptr && !deg[r]) q.push(r);
        }
    }

    void remove()
    {
        if (left != nullptr) {
            left->count--;
            if (!left->count) {
                left->remove();
                // std::cout << std::format("delete node {}\n", left->id());
                left = nullptr;
                delete left;
            }
        }
        if (right != nullptr) {
            right->count--;
            if (!right->count) {
                right->remove();
                // std::cout << std::format("delete node {}\n", right->id());
                delete right;
                right = nullptr;
            }
        }
        left = right = nullptr;
        op = ops::none;
    }

    void print()
    {
        std::cerr << std::format("{}", to_string()) << std::endl;
        if (left != nullptr) {
            std::cerr << std::format("{0} ---{2}--> {1}", left->id(), id(),
                                     op_name(op))
                      << std::endl;
            left->print();
        }
        if (right != nullptr) {
            std::cerr << std::format("{0} ---{2}--> {1}", right->id(), id(),
                                     op_name(op))
                      << std::endl;
            right->print();
        }
    }
};

template <typename T> class Variable
{
public:
    T value;
    TapeNode<T>* node;

    const T raw() const { return value; }
    const T diff() const { return node->diff; }
    T operator()() { return value; }

    Variable(T value = 0, bool require_diff = true)
        : value(value), node(new TapeNode<T>(value))
    {
        node->count++;
        // std::cerr << "created " << node->id() << ", " << this << std::endl;
        // if (node != nullptr)
        //     std::cerr << node->to_string() << std::endl;
        // else
        //     std::cerr << "node(nullptr)" << std::endl;
    }

    Variable(TapeNode<T>* node) : node(node), value(node->value)
    {
        node->count++;
        // std::cerr << "copied " << node->id() << ", " << this << std::endl;
        // if (node != nullptr)
        //     std::cerr << node->to_string() << std::endl;
        // else
        //     std::cerr << "node(nullptr)" << std::endl;
    }

    Variable(const Variable& v) : node(v.node), value(v.value)
    {
        node->count++;
        // std::cerr << "copied " << node->id() << ", " << this << std::endl;
        // if (node != nullptr)
        //     std::cerr << node->to_string() << std::endl;
        // else
        //     std::cerr << "node(nullptr)" << std::endl;
    }

    Variable(Variable&& v) : node(v.node), value(v.value)
    {
        v.node = nullptr;
        // std::cerr << "copied " << node->id() << ", " << this << std::endl;
        // std::cerr << node->to_string() << std::endl;
    }

    ~Variable()
    {
        // std::cerr << "-> destructing " << this << std::endl;
        if (node != nullptr) {
            node->count--;
            // std::cerr << std::format("try delete node {}, remaining {}\n",
            // node->id(), node->count);
            if (node->count == 0) {
                // std::cerr << std::format("delete node {}\n", node->id());
                node->remove();
                delete node;
                node = nullptr;
            }
            //} else {
            // std::cerr << std::format("try delete null node, value: {}\n",
            // value);
        }
        // std::cerr << "-> destructed." << std::endl;
    }

    Variable& operator=(const Variable& other)
    {
        if (this == &other) return *this;
        Variable::~Variable();
        node = other.node;
        value = other.value;
        node->count++;
        return *this;
    }

    Variable& operator=(Variable&& other)
    {
        if (this == &other) return *this;
        Variable::~Variable();
        node = other.node;
        value = other.value;
        other.node = nullptr;
        return *this;
    }

    bool operator==(const Variable& other) const
    {
        return abs(value - other.value) < 1e-10;
    }
    bool operator!=(const Variable& other) const
    {
        return abs(value - other.value) >= 1e-10;
    }
    bool operator<(const Variable& other) const { return value < other.value; }
    bool operator>(const Variable& other) const { return value > other.value; }
    bool operator<=(const Variable& other) const
    {
        return value <= other.value;
    }
    bool operator>=(const Variable& other) const
    {
        return value >= other.value;
    }

    void clear() { this->node->diff = 0; }
    void propagate(bool remain = false)
    {
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

    template <typename... Args> auto derivative(const Args&... args)
    {
        propagate();
        return std::make_tuple(args.diff()...);
    }

    friend std::ostream& operator<<(std::ostream& os, const Variable& v)
    {
        os << v.node->value;
        return os;
    }
    friend std::istream& operator>>(std::istream& os, const Variable& v)
    {
        os >> v.node->value;
        return os;
    }

    friend Variable operator+(const Variable& v) { return Variable(v); }
    friend Variable operator+(const Variable& a, const Variable& b)
    {
        return Variable(new TapeNode<T>(a.node->value + b.node->value, ops::add,
                                        a.node, b.node));
    }
    friend Variable operator-(const Variable& v)
    {
        return Variable(new TapeNode<T>(-v.node->value, ops::oppo, v.node));
    }
    friend Variable operator-(const Variable& a, const Variable& b)
    {
        return Variable(new TapeNode<T>(a.node->value - b.node->value, ops::sub,
                                        a.node, b.node));
    }
    friend Variable operator*(const Variable& a, const Variable& b)
    {
        return Variable(new TapeNode<T>(a.node->value * b.node->value, ops::mul,
                                        a.node, b.node));
    }
    friend Variable operator/(const Variable& a, const Variable& b)
    {
        return Variable(new TapeNode<T>(a.node->value / b.node->value,
                                        ops::divis, a.node, b.node));
    }
    friend Variable operator^(const Variable& a, const Variable& b)
    {
        return pow(a, b);
    }
    friend Variable log(const Variable& v)
    {
        return Variable(
            new TapeNode<T>(std::log(v.node->value), ops::ln, v.node, nullptr));
    }
    friend Variable sin(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::sin(v.node->value), ops::sine,
                                        v.node, nullptr));
    }
    friend Variable cos(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::cos(v.node->value), ops::cosine,
                                        v.node, nullptr));
    }
    friend Variable tan(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::tan(v.node->value), ops::tangent,
                                        v.node, nullptr));
    }
    friend Variable exp(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::exp(v.node->value), ops::expo,
                                        v.node, nullptr));
    }
    friend Variable sqrt(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::sqrt(v.node->value), ops::sqroot,
                                        v.node, nullptr));
    }
    friend Variable asin(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::asin(v.node->value), ops::arcsin,
                                        v.node, nullptr));
    }
    friend Variable acos(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::acos(v.node->value), ops::arccos,
                                        v.node, nullptr));
    }
    friend Variable atan(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::atan(v.node->value), ops::arctan,
                                        v.node, nullptr));
    }
    friend Variable pow(const Variable& a, const Variable& b)
    {
        return Variable(new TapeNode<T>(std::pow(a.node->value, b.node->value),
                                        ops::power, a.node, b.node));
    }
    friend Variable sinh(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::sinh(v.node->value), ops::sin_h,
                                        v.node, nullptr));
    }
    friend Variable cosh(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::cosh(v.node->value), ops::cos_h,
                                        v.node, nullptr));
    }
    friend Variable tanh(const Variable& v)
    {
        return Variable(new TapeNode<T>(std::tanh(v.node->value), ops::tan_h,
                                        v.node, nullptr));
    }
};

template <typename... Args> void clear_diff(Args... v) { (v.clear(), ...); }

template <typename T> struct std::formatter<Variable<T>> : std::formatter<T> {
    auto format(const Variable<T>& v, std::format_context& ctx) const
    {
        return std::formatter<T>::format(v.value, ctx);
    }
};

using var = Variable<double>;