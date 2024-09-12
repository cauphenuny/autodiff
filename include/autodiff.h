#pragma once
#include <cassert>
#include <format>
#include <iostream>

enum ops {
    none,
    eq,     // a = b
    oppo,   // -a
    plus,   // a + b
    minus,  // a - b
    mul,    // a * b
    div,    // a / b
    oplog,  // log(a)
    opsin,  // sin(a)
    opcos,  // cos(a)
    optan,  // tan(a)
};

const char* op_name(ops);

using value_type = double;
using size_type = int;

class tape_node
{
public:
    ops op;
    tape_node* left;
    tape_node* right;
    value_type value, dif;
    int refcount;

    tape_node(
        value_type value, ops op = none, tape_node* left = nullptr,
        tape_node* right = nullptr)
        : value(value), op(op), left(left), right(right), dif(0), refcount(0)
    {
        if (left != nullptr) left->refcount++;
        if (right != nullptr) right->refcount++;
    }

    void backpropagate();

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
    tape_node* node;
    value_type operator()() { return node->value; }
    value_type diff() { return node->dif; }

    variable(value_type value = 0) : node(new tape_node(value))
    {
        node->refcount++;
        //std::cerr << "created " << node->id() << std::endl;
        //if (node != nullptr)
        //    std::cerr << node->to_string() << std::endl;
        //else
        //    std::cerr << "node(nullptr)" << std::endl;
    }

    variable(tape_node* node) : node(node)
    {
        node->refcount++;
        //std::cerr << "created " << node->id() << std::endl;
        //if (node != nullptr)
        //    std::cerr << node->to_string() << std::endl;
        //else
        //    std::cerr << "node(nullptr)" << std::endl;
    }

    variable(const variable& var) : node(var.node)
    {
        node->refcount++;
        // node = new tape_node(var.node->value, ops::eq, var.node);
        //if (node != nullptr)
        //    std::cerr << "added " << node->id() << std::endl;
        //else
        //    std::cerr << "node(nullptr)" << std::endl;
    }

    variable(variable&& var) : node(var.node)
    {
        var.node = nullptr;
        std::cerr << "moved" << std::endl;
    }

    ~variable()
    {
        assert(node != nullptr);
        node->refcount--;
        //std::cerr << std::format(
        //    "try delete node {}, remaining {}\n", node->id(), node->refcount);
        if (node->refcount == 0) {
            //std::cerr << std::format("delete node {}\n", node->id());
            node->remove();
            delete node;
            node = nullptr;
        }
    }

    variable& operator=(const variable& other)
    {
        if (this == &other) return *this;
        variable::~variable();
        node = other.node;
        node->refcount++;
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
    friend variable log(const variable& var);
    friend variable sin(const variable& var);
    friend variable cos(const variable& var);
    friend variable tan(const variable& var);
};
