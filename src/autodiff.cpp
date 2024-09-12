// author: cauphenuny
// date: 2024/09/11
#include "autodiff.h"

#include "util.h"

#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <istream>
#include <map>
#include <queue>

void tape_node::print()
{
    std::cerr << id() << "," << refcount << std::endl;
    if (left != nullptr) {
        std::cerr << std::format(
                         "{},{} {},{} {}", left->id(), left->refcount, id(),
                         refcount, op_name(op))
                  << std::endl;
        left->print();
    }
    if (right != nullptr) {
        std::cerr << std::format(
                         "{},{} {},{} {}", right->id(), right->refcount, id(),
                         refcount, op_name(op))
                  << std::endl;
        right->print();
    }
}

void tape_node::remove()
{
    if (left != nullptr) {
        left->refcount--;
        if (!left->refcount) {
            left->remove();
            //std::cout << std::format("delete node {}\n", left->id());
            delete left;
        }
    }
    if (right != nullptr) {
        right->refcount--;
        if (!right->refcount) {
            right->remove();
            //std::cout << std::format("delete node {}\n", right->id());
            delete right;
        }
    }
}

void tape_node::backpropagate()
{
    std::map<tape_node*, int> deg;
    std::function<void(tape_node*)> find = [&](tape_node* var) {
        if (deg.count(var)) return;
        deg[var] = 0;
        auto l = var->left, r = var->right;
        if (l != nullptr) find(l), deg[l]++;
        if (r != nullptr) find(r), deg[r]++;
    };
    find(this);
    //for (auto [p, cnt] : deg) {
    //    std::cerr << std::format("selected: {}, refcnt: {}", p->id(), cnt)
    //              << std::endl;
    //}
    std::queue<tape_node*> q;
    this->dif = 1;
    q.push(this);
    while (q.size()) {
        tape_node* cur = q.front();
        //debugln(cur->to_string());
        q.pop();
        tape_node *l = cur->left, *r = cur->right;
        switch (cur->op) {
            case ops::none: continue;
            case ops::eq: deg[l]--, l->dif += cur->dif; break;
            case ops::oppo: deg[l]--, l->dif -= cur->dif; break;
            case ops::plus:
                deg[l]--, deg[r]--;
                l->dif += cur->dif;
                r->dif += cur->dif;
                break;
            case ops::minus:
                deg[l]--, deg[r]--;
                l->dif += cur->dif;
                r->dif -= cur->dif;
                break;
            case ops::mul:
                deg[l]--, deg[r]--;
                l->dif += cur->dif * r->value;
                r->dif += cur->dif * l->value;
                break;
            case ops::div:
                deg[l]--, deg[r]--;
                l->dif += cur->dif / r->value;
                r->dif -= cur->dif * l->value / (r->value * r->value);
                break;
            case ops::oplog:
                deg[l]--;
                l->dif += cur->dif / l->value;
                break;
            case ops::opsin:
                deg[l]--;
                l->dif += cur->dif * std::cos(l->value);
                break;
            case ops::opcos:
                deg[l]--;
                l->dif -= cur->dif * std::sin(l->value);
                break;
            case ops::optan:
                deg[l]--;
                l->dif += cur->dif / (std::cos(l->value) * std::cos(l->value));
                break;
        }
        //std::cerr << std::format("left: {}, refcnt: {}\n", l->id(), deg[l]);
        //std::cerr << std::format("right: {}, refcnt: {}\n", r->id(), deg[r]);
        if (l != nullptr && !deg[l]) q.push(l);
        if (r != l && r != nullptr && !deg[r]) q.push(r);
    }
}

const char* op_name(ops id)
{
    switch (id) {
        case ops::none: return "none ";
        case ops::eq: return "eq   ";
        case ops::oppo: return "oppo ";
        case ops::plus: return "plus ";
        case ops::minus: return "minus";
        case ops::mul: return "mul  ";
        case ops::div: return "div  ";
        case ops::oplog: return "log  ";
        case ops::opsin: return "sin  ";
        case ops::opcos: return "cos  ";
        case ops::optan: return "tan  ";
    }
    return "unknown";
}
std::string tape_node::id() const
{
    return std::format("#{:02X}", (((size_t)this) & 0xfff) >> 4);
}

std::string tape_node::to_string() const
{
    return std::format(
        "node(id: {}, op: {}, l/r: {} / {}, value: {}, dif: {})", this->id(),
        op_name(op), left ? left->id() : "   ", right ? right->id() : "   ",
        value, dif);
}

std::ostream& operator<<(std::ostream& os, const tape_node& v)
{
    os << v.to_string();
    return os;
}
std::istream& operator>>(std::istream& is, const tape_node& v)
{
    is >> v.value;
    return is;
}

std::ostream& operator<<(std::ostream& os, const variable& v)
{
    os << v.node->value;
    return os;
}
std::istream& operator>>(std::istream& os, const variable& v)
{
    os >> v.node->value;
    return os;
}

variable operator+(variable a, variable b)
{
    return variable(new tape_node(
        a.node->value + b.node->value, ops::plus, a.node, b.node));
}
variable operator-(variable v)
{
    return variable(new tape_node(-v.node->value, ops::oppo, v.node));
}
variable operator-(variable a, variable b)
{
    return variable(new tape_node(
        a.node->value - b.node->value, ops::minus, a.node, b.node));
}
variable operator*(variable a, variable b)
{
    return variable(
        new tape_node(a.node->value * b.node->value, ops::mul, a.node, b.node));
}
variable operator/(variable a, variable b)
{
    return variable(
        new tape_node(a.node->value / b.node->value, ops::div, a.node, b.node));
}
variable log(const variable& var)
{
    return variable(new tape_node(
        std::log(var.node->value), ops::oplog, var.node, nullptr));
}
variable sin(const variable& var)
{
    return variable(new tape_node(
        std::sin(var.node->value), ops::opsin, var.node, nullptr));
}
variable cos(const variable& var)
{
    return variable(new tape_node(
        std::cos(var.node->value), ops::opcos, var.node, nullptr));
}
variable tan(const variable& var)
{
    return variable(new tape_node(
        std::tan(var.node->value), ops::optan, var.node, nullptr));
}