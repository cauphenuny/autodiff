#pragma once
#include <iostream>
#include <cassert>

enum ops {
    none,
    eq,     // a = b
    oppo,   // -a
    plus,   // a + b
    minus,  // a - b
    mul,    // a * b
    div,    // a / b
};

using value_type = double;
using size_type = int;

class tape_node
{
private:
    ops op;
    tape_node* left;
    tape_node* right;

public:
    value_type value, dif;
    int refcount;

    tape_node(value_type value, ops op = none, tape_node* left = nullptr,
              tape_node* right = nullptr)
        : value(value), op(op), left(left), right(right), dif(0), refcount(1)
    {
        if (left != nullptr) left->refcount++;
        if (right != nullptr) right->refcount++;
    }

    void backpropagate();

    void remove();
    friend std::ostream& operator<<(std::ostream& os, const tape_node& v);
    friend std::istream& operator>>(std::istream& os, const tape_node& v);
};

class variable
{
private:
    tape_node* node;

public:
    value_type operator()() { return node->value; }
    value_type diff() { return node->dif; }

    variable(value_type value)
    {
        node = new tape_node(value);
    }

    variable(tape_node* node) : node(node) {}

    variable(const variable& var)
    {
        node = new tape_node(var.node->value, ops::eq, var.node);
    }

    variable(variable&& var) : node(var.node) { var.node = nullptr; }

    ~variable()
    {
        assert(node != nullptr);
        node->refcount--;
        if (node->refcount == 0) {
            node->remove();
            delete node;
            node = nullptr;
        }
    }

    variable& operator=(const variable& other)
    {
        if (this == &other) return *this;
        variable::~variable();
        node = new tape_node(other.node->value, ops::eq, other.node);
        return *this;
    }

    friend variable operator+(variable left, variable right);
    friend variable operator-(variable var);
    friend variable operator-(variable left, variable right);
    friend variable operator*(variable left, variable right);
    friend variable operator*(variable var, value_type scalar);
    friend variable operator/(variable left, variable right);

    void clear() { this->node->dif = 0; }
    void backpropagate() { node->backpropagate(); }

    friend std::ostream& operator<<(std::ostream& os, const variable& v);
    friend std::istream& operator>>(std::istream& is, const variable& v);
};

variable log(const variable &var);
variable sin(const variable &var);
variable cos(const variable &var);
variable tan(const variable &var);