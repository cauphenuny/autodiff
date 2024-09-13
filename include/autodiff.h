#pragma once
#include <array>
#include <cassert>
#include <format>
#include <iostream>

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

constexpr const char* op_name(ops);

class tape_node
{
public:
    using value_type = double;
    using size_type = int;
    ops op{none};
    tape_node* left{nullptr};
    tape_node* right{nullptr};
    value_type value{0}, diff{0};
    int count{0};  // reference count
    bool require_diff{true};

    tape_node(value_type value, ops op = none, tape_node* left = nullptr,
              tape_node* right = nullptr)
        : value(value), op(op), left(left), right(right), diff(0), count(0)
    {
        if (left != nullptr) left->count++;
        if (right != nullptr) right->count++;
    }

    void propagate();

    void remove();

    void print();

    std::string id() const;
    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& os, const tape_node& v);
    friend std::istream& operator>>(std::istream& os, const tape_node& v);
};

class var
{
private:
public:
    using value_type = tape_node::value_type;
    using size_type = int;
    value_type value;
    tape_node* node;
    const value_type raw() const { return node->value; }
    value_type operator()() { return value; }
    const value_type diff() const { return node->diff; }

    var(value_type value = 0, bool require_diff = true)
        : value(value), node(new tape_node(value))
    {
        node->count++;
        // std::cerr << "created " << node->id() << std::endl;
        // if (node != nullptr)
        //     std::cerr << node->to_string() << std::endl;
        // else
        //     std::cerr << "node(nullptr)" << std::endl;
    }

    var(tape_node* node) : node(node), value(node->value)
    {
        node->count++;
        // std::cerr << "created " << node->id() << std::endl;
        // if (node != nullptr)
        //     std::cerr << node->to_string() << std::endl;
        // else
        //     std::cerr << "node(nullptr)" << std::endl;
    }

    var(const var& v) : node(v.node), value(v.value)
    {
        node->count++;
        // node = new tape_node(v.node->value, ops::eq, v.node);
        // if (node != nullptr)
        //    std::cerr << "added " << node->id() << std::endl;
        // else
        //    std::cerr << "node(nullptr)" << std::endl;
    }

    var(var&& v) : node(v.node), value(v.value)
    {
        v.node = nullptr;
        // std::cerr << "moved" << std::endl;
    }

    ~var()
    {
        if (node != nullptr) {
            node->count--;
            // std::cerr << std::format(
            //     "try delete node {}, remaining {}\n", node->id(),
            //     node->count);
            if (node->count == 0) {
                // std::cerr << std::format("delete node {}\n", node->id());
                node->remove();
                delete node;
                node = nullptr;
            }
        }
    }

    var& operator=(const var& other)
    {
        if (this == &other) return *this;
        var::~var();
        node = other.node;
        value = other.value;
        node->count++;
        return *this;
    }

    var& operator=(var&& other)
    {
        if (this == &other) return *this;
        var::~var();
        node = other.node;
        value = other.value;
        other.node = nullptr;
        return *this;
    }

    bool operator==(const var& other) const
    {
        return abs(value - other.value) < 1e-10;
    }
    bool operator!=(const var& other) const
    {
        return abs(value - other.value) >= 1e-10;
    }
    bool operator<(const var& other) const { return value < other.value; }
    bool operator>(const var& other) const { return value > other.value; }
    bool operator<=(const var& other) const { return value <= other.value; }
    bool operator>=(const var& other) const { return value >= other.value; }

    void clear() { this->node->diff = 0; }
    void propagate(bool remain = false)
    {
        if (node == nullptr) {
            std::cerr << "propagate nullptr" << std::endl;
            return;
        }
        node->propagate();
        if (!remain) {
            var::~var();
        }
    }
    void require_diff(bool require_diff) { node->require_diff = require_diff; }

    template <typename... Args> auto derivative(const Args&... args)
    {
        propagate();
        return std::make_tuple(args.diff()...);
    }

    friend var operator+(const var& v);
    friend var operator+(const var& left, const var& right);
    friend var operator-(const var& v);
    friend var operator-(const var& left, const var& right);
    friend var operator*(const var& left, const var& right);
    friend var operator/(const var& left, const var& right);
    friend var operator^(const var& left, const var& right);

    friend std::ostream& operator<<(std::ostream& os, const var& v);
    friend std::istream& operator>>(std::istream& is, const var& v);
    friend var sqrt(const var& v);
    friend var log(const var& v);
    friend var exp(const var& v);
    friend var sin(const var& v);
    friend var cos(const var& v);
    friend var tan(const var& v);
    friend var asin(const var& v);
    friend var acos(const var& v);
    friend var atan(const var& v);
    friend var pow(const var& a, const var& b);
    friend var abs(const var& v);
    friend var pow(const var& v);
    friend var sinh(const var& v);
    friend var cosh(const var& v);
    friend var tanh(const var& v);
};

template <typename... Args> void clear_diff(Args... v) { (v.clear(), ...); }

template <> struct std::formatter<var> : std::formatter<var::value_type> {
    auto format(const var& v, std::format_context& ctx) const
    {
        return std::formatter<var::value_type>::format(v.value, ctx);
    }
};