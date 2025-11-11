#pragma once
#include "util.hpp"

#include <format>
#include <iostream>
#include <istream>
#include <map>
#include <queue>

template <typename T> class Operation {
public:
    enum class OpType { unknown, unary, binary };
    OpType op_type{OpType::unknown};
    virtual std::string_view name() const = 0;
    virtual T backward(const T& diff, const T& arg) const {
        runtimeError("Not implemented");
    }
    virtual std::tuple<T, T> backward(const T& diff, const T& lhs, const T& rhs) const {
        runtimeError("Not implemented");
    }
    virtual T forward(const T& arg) const { runtimeError("Not implemented"); }
    virtual T forward(const T& lhs, const T& rhs) const {
        runtimeError("Not implemented");
    }
};

template <typename T> class TapeNode {
    const Operation<T>* const op{nullptr};
    TapeNode* lhs{nullptr};
    TapeNode* rhs{nullptr};
    T _value{0}, _diff{0};
    int _ref_count{0};
    bool _require_diff{true};

public:
    int ref_count() { return _ref_count; }
    void add_ref() { _ref_count++; }
    void remove_ref() {
        _ref_count--;
        if (_ref_count == 0) {
            remove();
        }
    }

    explicit TapeNode<T>(T value, Operation<T>* oper = nullptr,
                         TapeNode<T>* left = nullptr, TapeNode<T>* right = nullptr)
        : op(std::move(oper)), lhs(left), rhs(right), _value(value) {
        if (left != nullptr) left->_ref_count++;
        if (right != nullptr) right->_ref_count++;
        this->_ref_count = 1;
    }

    T& value() { return _value; }
    const T& value() const { return _value; }
    T diff() { return _diff; }
    void clear() { _diff = 0; }
    void require_diff(bool require_diff) { _require_diff = require_diff; }

    std::string id() const { return std::format("#{:02X}", ((size_t)this & 0xfff) >> 4); }

    std::string name() const {
        return std::format("#{:02X}:{:.4}:{}/{}", ((size_t)this & 0xfff) >> 4, _value,
                           _diff, _ref_count);
    }

    std::string to_string() const {
        return std::format("node(id: {}, func: {}, l/r: {}/{}, v: {}, d: {}, ref: {})",
                           this->id(), op->name(), lhs ? lhs->id() : "   ",
                           rhs ? rhs->id() : "   ", _value, _diff, _ref_count);
    }

    friend std::ostream& operator<<(std::ostream& os, const TapeNode& v) {
        os << v.to_string();
        return os;
    }
    friend std::istream& operator>>(std::istream& is, TapeNode& v) {
        is >> v._value;
        return is;
    }

    void propagate(T initial_diff) {
        std::map<TapeNode*, int> deg;
        auto find = [&](auto&& find, TapeNode* v) -> void {
            if (deg.count(v)) return;
            deg[v] = 0;
            auto l = v->lhs, r = v->rhs;
            if (l != nullptr) find(find, l), deg[l]++;
            if (r != nullptr) find(find, r), deg[r]++;
        };
        find(find, this);
        std::queue<TapeNode*> q;
        this->_diff = initial_diff;
        q.push(this);
        while (q.size()) {
            TapeNode* cur = q.front();
            q.pop();
            TapeNode *l = cur->lhs, *r = cur->rhs;
            if (!cur->op) continue;
            switch (cur->op->op_type) {
                case Operation<T>::OpType::unary:
                    deg[l]--, l->_diff += cur->op->backward(cur->_diff, l->_value);
                    break;
                case Operation<T>::OpType::binary: {
                    deg[l]--, deg[r]--;
                    auto [dl, dr] = cur->op->backward(cur->_diff, l->_value, r->_value);
                    l->_diff += dl, r->_diff += dr;
                    break;
                }
                default: runtimeError("invalid op type");
            }
            if (l != nullptr && !deg[l]) q.push(l);
            if (r != nullptr && r != l && !deg[r]) q.push(r);
        }
    }

    void remove() {
        for (auto& child : {lhs, rhs}) {
            if (!child) continue;
            child->remove_ref();
            if (!child->ref_count()) {
                delete child;
            }
        }
        lhs = rhs = nullptr;
    }

    void print() {
        std::cerr << std::format("{}", to_string()) << std::endl;
        for (auto child : {lhs, rhs}) {
            if (child) {
                std::cerr << std::format("{0} ---{2}--> {1}\n", child->id(), id(),
                                         op->name());
                child->print();
            }
        }
    }
};

template <typename T> class AutoDiff {
    void delete_node() {
        if (node != nullptr) {
            node->remove_ref();
            if (!node->ref_count()) delete node;
            node = nullptr;
        }
    }

public:
    TapeNode<T>* node;
    const T& raw() const { return node->value(); }
    T& raw() { return node->value(); }
    T diff() const { return node->diff(); }
    virtual T initial_diff() const = 0;
    void clear() { this->node->clear(); }

    AutoDiff<T>(T value) : node(new TapeNode<T>(value)) {}
    ~AutoDiff<T>() { delete_node(); }
    AutoDiff<T>(const AutoDiff<T>& other) : node(other.node) { node->add_ref(); }
    AutoDiff<T>(AutoDiff<T>&& other) noexcept : node(other.node) { other.node = nullptr; }
    AutoDiff<T>& operator=(const AutoDiff<T>& other) {
        if (this == &other) return *this;
        delete_node();
        node = other.node;
        node->add_ref();
        return *this;
    }
    AutoDiff<T>& operator=(AutoDiff<T>&& other) noexcept {
        if (this == &other) return *this;
        delete_node();
        node = other.node;
        other.node = nullptr;
        return *this;
    }
    AutoDiff<T>(Operation<T>* op, const auto&... args) {
        node = new TapeNode<T>(op->forward((args.raw())...), op, (args.node)...);
    }

    void propagate(bool remain_graph = false) {
        if (node == nullptr) {
            runtimeError("propagate nullptr");
        }
        node->propagate(initial_diff());
        if (!remain_graph) {
            node->remove();
        }
    }
    void require_diff(bool require_diff) { node->require_diff(require_diff); }

    template <typename... Args> auto derivative(const Args&... args) {
        propagate();
        return std::make_tuple(args.diff()...);
    }
};
