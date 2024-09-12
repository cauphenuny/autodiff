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
    std::cerr << id() << "," << count << std::endl;
    if (left != nullptr) {
        std::cerr << std::format(
                         "{},{} {},{} {}", left->id(), left->count, id(), count,
                         op_name(op))
                  << std::endl;
        left->print();
    }
    if (right != nullptr) {
        std::cerr << std::format(
                         "{},{} {},{} {}", right->id(), right->count, id(),
                         count, op_name(op))
                  << std::endl;
        right->print();
    }
}

void tape_node::remove()
{
    if (left != nullptr) {
        left->count--;
        if (!left->count) {
            left->remove();
            // std::cout << std::format("delete node {}\n", left->id());
            delete left;
        }
    }
    if (right != nullptr) {
        right->count--;
        if (!right->count) {
            right->remove();
            // std::cout << std::format("delete node {}\n", right->id());
            delete right;
        }
    }
}

void tape_node::propagate()
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
    // for (auto [p, cnt] : deg) {
    //     std::cerr << std::format("selected: {}, refcnt: {}", p->id(), cnt)
    //               << std::endl;
    // }
    std::queue<tape_node*> q;
    this->diff = 1;
    q.push(this);
    while (q.size()) {
        tape_node* cur = q.front();
        // debugln(cur->to_string());
        q.pop();
        // if (!cur->require_diff) continue;
        tape_node *l = cur->left, *r = cur->right;
        switch (cur->op) {
            case ops::none: continue;
            case ops::oppo: deg[l]--, l->diff -= cur->diff; break;
            case ops::plus:
                deg[l]--, deg[r]--;
                l->diff += cur->diff;
                r->diff += cur->diff;
                break;
            case ops::minus:
                deg[l]--, deg[r]--;
                l->diff += cur->diff;
                r->diff -= cur->diff;
                break;
            case ops::mul:
                deg[l]--, deg[r]--;
                l->diff += cur->diff * r->value;
                r->diff += cur->diff * l->value;
                break;
            case ops::div:
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
        // std::cerr << std::format("left: {}, refcnt: {}\n", l->id(), deg[l]);
        // std::cerr << std::format("right: {}, refcnt: {}\n", r->id(), deg[r]);
        if (l != nullptr && !deg[l]) q.push(l);
        if (r != l && r != nullptr && !deg[r]) q.push(r);
    }
}

constexpr const char* op_name(ops id)
{
    switch (id) {
        case ops::none: return "none ";
        case ops::oppo: return "oppo ";
        case ops::plus: return "plus ";
        case ops::minus: return "minus";
        case ops::mul: return "mul  ";
        case ops::div: return "div  ";
        case ops::ln: return "log  ";
        case ops::expo: return "exp  ";
        case ops::sine: return "sin  ";
        case ops::cosine: return "cos  ";
        case ops::tangent: return "tan  ";
        case ops::arcsin: return "asin ";
        case ops::arccos: return "acos ";
        case ops::arctan: return "atan ";
        case ops::abso: return "abs  ";
        case ops::power: return "pow  ";
        case ops::sqroot: return "sqrt ";
        case ops::sin_h: return "sinh ";
        case ops::cos_h: return "cosh ";
        case ops::tan_h: return "tanh ";
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
        "node(id: {}, op: {}, l/r: {} / {}, value: {}, diff: {})", this->id(),
        op_name(op), left ? left->id() : "   ", right ? right->id() : "   ",
        value, diff);
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

variable operator+(const variable& v) { return variable(v); }
variable operator+(const variable& a, const variable& b)
{
    return variable(new tape_node(
        a.node->value + b.node->value, ops::plus, a.node, b.node));
}
variable operator-(const variable& v)
{
    return variable(new tape_node(-v.node->value, ops::oppo, v.node));
}
variable operator-(const variable& a, const variable& b)
{
    return variable(new tape_node(
        a.node->value - b.node->value, ops::minus, a.node, b.node));
}
variable operator*(const variable& a, const variable& b)
{
    return variable(
        new tape_node(a.node->value * b.node->value, ops::mul, a.node, b.node));
}
variable operator/(const variable& a, const variable& b)
{
    return variable(
        new tape_node(a.node->value / b.node->value, ops::div, a.node, b.node));
}
variable operator^(const variable& a, const variable& b) { return pow(a, b); }
variable log(const variable& var)
{
    return variable(
        new tape_node(std::log(var.node->value), ops::ln, var.node, nullptr));
}
variable sin(const variable& var)
{
    return variable(
        new tape_node(std::sin(var.node->value), ops::sine, var.node, nullptr));
}
variable cos(const variable& var)
{
    return variable(new tape_node(
        std::cos(var.node->value), ops::cosine, var.node, nullptr));
}
variable tan(const variable& var)
{
    return variable(new tape_node(
        std::tan(var.node->value), ops::tangent, var.node, nullptr));
}
variable exp(const variable& var)
{
    return variable(
        new tape_node(std::exp(var.node->value), ops::expo, var.node, nullptr));
}
variable sqrt(const variable& var)
{
    return variable(new tape_node(
        std::sqrt(var.node->value), ops::sqroot, var.node, nullptr));
}
variable asin(const variable& var)
{
    return variable(new tape_node(
        std::asin(var.node->value), ops::arcsin, var.node, nullptr));
}
variable acos(const variable& var)
{
    return variable(new tape_node(
        std::acos(var.node->value), ops::arccos, var.node, nullptr));
}
variable atan(const variable& var)
{
    return variable(new tape_node(
        std::atan(var.node->value), ops::arctan, var.node, nullptr));
}
variable pow(const variable& a, const variable& b)
{
    return variable(new tape_node(
        std::pow(a.node->value, b.node->value), ops::power, a.node, b.node));
}
variable sinh(const variable& var)
{
    return variable(new tape_node(
        std::sinh(var.node->value), ops::sin_h, var.node, nullptr));
}
variable cosh(const variable& var)
{
    return variable(new tape_node(
        std::cosh(var.node->value), ops::cos_h, var.node, nullptr));
}
variable tanh(const variable& var)
{
    return variable(new tape_node(
        std::tanh(var.node->value), ops::tan_h, var.node, nullptr));
}