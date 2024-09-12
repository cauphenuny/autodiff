#pragma once
#include <cassert>
#include <format>
#include <iostream>

enum ops {
    none,
    oppo,     // -a
    plus,     // a + b
    minus,    // a - b
    mul,      // a * b
    div,      // a / b
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

using value_type = double;
using size_type = int;

class tape_node
{
public:
    ops op;
    tape_node* left;
    tape_node* right;
    value_type value, diff;
    int count;  // reference count
    // bool require_diff;

    tape_node(
        value_type value, ops op = none, tape_node* left = nullptr,
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

class variable
{
private:
public:
    value_type value;
    tape_node* node;
    value_type operator()() { return value; }
    value_type diff() { return node->diff; }

    variable(value_type value = 0, bool require_diff = true)
        : value(value), node(new tape_node(value))
    {
        node->count++;
        // std::cerr << "created " << node->id() << std::endl;
        // if (node != nullptr)
        //     std::cerr << node->to_string() << std::endl;
        // else
        //     std::cerr << "node(nullptr)" << std::endl;
    }

    variable(tape_node* node) : node(node), value(node->value)
    {
        node->count++;
        // std::cerr << "created " << node->id() << std::endl;
        // if (node != nullptr)
        //     std::cerr << node->to_string() << std::endl;
        // else
        //     std::cerr << "node(nullptr)" << std::endl;
    }

    variable(const variable& var) : node(var.node), value(var.value)
    {
        node->count++;
        // node = new tape_node(var.node->value, ops::eq, var.node);
        // if (node != nullptr)
        //    std::cerr << "added " << node->id() << std::endl;
        // else
        //    std::cerr << "node(nullptr)" << std::endl;
    }

    variable(variable&& var) : node(var.node), value(var.value)
    {
        var.node = nullptr;
        // std::cerr << "moved" << std::endl;
    }

    ~variable()
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

    variable& operator=(const variable& other)
    {
        if (this == &other) return *this;
        variable::~variable();
        node = other.node;
        node->count++;
        return *this;
    }

    friend variable operator+(variable var);
    friend variable operator+(variable left, variable right);
    friend variable operator-(variable var);
    friend variable operator-(variable left, variable right);
    friend variable operator*(variable left, variable right);
    friend variable operator*(variable var, value_type scalar);
    friend variable operator/(variable left, variable right);

    void clear() { this->node->diff = 0; }
    void propagate(bool remain = false)
    {
        if (node == nullptr) {
            std::cerr << "propagate nullptr" << std::endl;
            return;
        }
        node->propagate();
        if (!remain) {
            variable::~variable();
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const variable& v);
    friend std::istream& operator>>(std::istream& is, const variable& v);
    friend variable sqrt(const variable& var);
    friend variable log(const variable& var);
    friend variable exp(const variable& var);
    friend variable sin(const variable& var);
    friend variable cos(const variable& var);
    friend variable tan(const variable& var);
    friend variable asin(const variable& var);
    friend variable acos(const variable& var);
    friend variable atan(const variable& var);
    friend variable pow(const variable& a, const variable& b);
    friend variable abs(const variable& var);
    friend variable pow(const variable& var);
    friend variable sinh(const variable& var);
    friend variable cosh(const variable& var);
    friend variable tanh(const variable& var);
};
