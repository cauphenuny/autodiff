#pragma once
#include "util.hpp"

#include <format>
#include <iostream>
#include <istream>
#include <map>
#include <queue>

template <typename T> class AutoDiff {
public:
    virtual void clear() = 0;
    virtual void propagate(bool remain_graph) = 0;
    virtual const T& raw() const = 0;
    virtual T& raw() = 0;
    virtual T diff() const = 0;
    virtual T initial_diff() const = 0;
};

template <typename T> class Operation {
public:
    enum class OpType { unknown, none, unary, binary, ternary };
    OpType op_type{OpType::unknown};
    virtual std::string_view name() const = 0;
    virtual T backward(const T& diff, const T& arg) const {
        runtimeError("Not implemented");
    }
    virtual std::tuple<T, T> backward(const T& diff, const T& lhs, const T& rhs) const {
        runtimeError("Not implemented");
    }
    virtual std::tuple<T, T, T> backward(const T& diff, const T& arg1, const T& arg2,
                                         const T& arg3) const {
        runtimeError("Not implemented");
    }
};

template <typename T> class TapeNode {
    const Operation<T>* const op{nullptr};
    TapeNode* first{nullptr};
    TapeNode* second{nullptr};
    TapeNode* third{nullptr};
    T _value{0}, _diff{0};
    int _ref_count{0};
    bool _require_diff{true};

public:
    explicit TapeNode<T>(T value, Operation<T>* oper, TapeNode<T>* left = nullptr,
                         TapeNode<T>* right = nullptr, TapeNode<T>* third = nullptr)
        : op(std::move(oper)), first(left), second(right), third(third), _value(value) {
        if (left != nullptr) left->_ref_count++;
        if (right != nullptr) right->_ref_count++;
        this->_ref_count = 1;
    }

    T& value() { return _value; }
    const T& value() const { return _value; }
    T diff() { return _diff; }
    void clear() { _diff = 0; }
    void require_diff(bool require_diff) { _require_diff = require_diff; }

    int ref_count() { return _ref_count; }
    void add_ref() { _ref_count++; }
    void remove_ref() {
        _ref_count--;
        if (_ref_count == 0) {
            remove();
        }
    }

    std::string id() const { return std::format("#{:02X}", ((size_t)this & 0xfff) >> 4); }

    std::string name() const {
        return std::format("#{:02X}:{:.4}:{}/{}", ((size_t)this & 0xfff) >> 4, _value,
                           _diff, _ref_count);
    }

    std::string to_string() const {
        return std::format("node(id: {}, func: {}, l/r: {}/{}, v: {}, d: {}, ref: {})",
                           this->id(), op->name(), first ? first->id() : "   ",
                           second ? second->id() : "   ", _value, _diff, _ref_count);
    }

    friend std::ostream& operator<<(std::ostream& os, const TapeNode& v) {
        os << v.to_string();
        return os;
    }
    friend std::istream& operator>>(std::istream& is, TapeNode& v) {
        is >> v._value;
        return is;
    }

    void propagate() {
        std::map<TapeNode*, int> deg;
        auto find = [&](auto&& find, TapeNode* v) -> void {
            if (deg.count(v)) return;
            deg[v] = 0;
            auto l = v->first, r = v->second;
            if (l != nullptr) find(find, l), deg[l]++;
            if (r != nullptr) find(find, r), deg[r]++;
        };
        find(find, this);
        std::queue<TapeNode*> q;
        this->_diff = 1;
        q.push(this);
        while (q.size()) {
            TapeNode* cur = q.front();
            q.pop();
            TapeNode *l = cur->first, *r = cur->second, *t = cur->third;
            switch (cur->op->op_type) {
                case Operation<T>::OpType::none: break;
                case Operation<T>::OpType::unary:
                    deg[l]--, l->_diff += cur->op->backward(cur->_diff, l->_value);
                    break;
                case Operation<T>::OpType::binary: {
                    deg[l]--, deg[r]--;
                    auto [dl, dr] = cur->op->backward(cur->_diff, l->_value, r->_value);
                    l->_diff += dl, r->_diff += dr;
                    break;
                }
                case Operation<T>::OpType::ternary: {
                    deg[l]--, deg[r]--, deg[t]--;
                    auto [dl, dr, dt] =
                        cur->op->backward(cur->_diff, l->_value, r->_value, t->_value);
                    l->_diff += dl, r->_diff += dr, t->_diff += dt;
                    break;
                }
                default: runtimeError("invalid op type");
            }
            if (l != nullptr && !deg[l]) q.push(l);
            if (r != nullptr && r != l && !deg[r]) q.push(r);
            if (t != nullptr && t != l && t != r && !deg[t]) q.push(t);
        }
    }

    void remove() {
        for (auto child : {first, second, third}) {
            if (!child) continue;
            child->remove_ref();
            if (!child->ref_count()) {
                delete child;
            }
        }
        first = second = third = nullptr;
    }

    void print() {
        std::cerr << std::format("{}", to_string()) << std::endl;
        for (auto child : {first, second, third}) {
            if (child) {
                std::cerr << std::format("{0} ---{2}--> {1}\n", child->id(), id(),
                                         op->name());
                child->print();
            }
        }
    }
};
