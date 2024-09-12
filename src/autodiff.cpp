// author: cauphenuny
// date: 2024/09/11
#include "autodiff.h"

#include "util.h"

#include <cassert>
#include <format>
#include <functional>
#include <istream>
#include <map>
#include <queue>

void tape_node::remove()
{
    if (left != nullptr) {
        left->refcount--;
        if (!left->refcount) {
            left->remove();
            delete left;
        }
    }
    if (right != nullptr) {
        right->refcount--;
        if (!right->refcount) {
            right->remove();
            delete right;
        }
    }
}

void tape_node::backpropagate()
{
    std::map<tape_node*, int> deg;
    std::function<void(tape_node*)> find = [&](tape_node* var) {
        if (!deg.count(var)) deg[var] = 0;
        auto l = var->left, r = var->right;
        if (l != nullptr) find(l), deg[l]++;
        if (r != nullptr) find(r), deg[r]++;
    };
    find(this);
    std::queue<tape_node*> q;
    this->dif = 1;
    q.push(this);
    while (q.size()) {
        tape_node* cur = q.front();
        //debugln(*cur);
        q.pop();
        tape_node *l = cur->left, *r = cur->right;
        switch (cur->op) {
            case ops::none: continue;
            case ops::eq:
                deg[l]--, l->dif += cur->dif;
                break;
            case ops::oppo:
                deg[l]--, l->dif -= cur->dif;
                break;
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
        }
        if (l != nullptr && !deg[l]) q.push(l);
        if (r != l && r != nullptr && !deg[r]) q.push(r);
    }
}

const char* id2name(ops id)
{
    switch (id) {
        case ops::none: return "none";
        case ops::eq: return "eq";
        case ops::oppo: return "oppo";
        case ops::plus: return "plus";
        case ops::minus: return "minus";
        case ops::mul: return "mul";
        case ops::div: return "div";
    }
    return "unknown";
}

std::ostream& operator<<(std::ostream& os, const tape_node& v)
{
    os << std::format(
        "id: {}, op: {}, l/r: {}/{}, value: {}, dif: {}", (void*)&v, id2name(v.op), (void*)v.left,
        (void*)v.right, v.value, v.dif);
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
    return variable(new tape_node(
        a.node->value * b.node->value, ops::mul, a.node, b.node));
}
variable operator/(variable a, variable b)
{
    return variable(new tape_node(
        a.node->value / b.node->value, ops::div, a.node, b.node));
}