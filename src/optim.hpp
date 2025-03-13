#pragma once
#include "propagate.hpp"

#include <initializer_list>
#include <set>

namespace optim {
template <typename T> class Optimizer {
protected:
    std::set<AutoDiff<T>*> params;

private:
    void traverse(AutoDiff<T>* v) {
        if (params.count(v)) return;
        params.insert(v);
    }
    void traverse(auto container) {
        for (auto& v : container) {
            traverse(v);
        }
    }

public:
    Optimizer(std::initializer_list<AutoDiff<T>*> parameters) { traverse(parameters); }
};

template <typename T> class GradientDescent : public Optimizer<T> {
    T learning_rate;

public:
    GradientDescent(std::initializer_list<AutoDiff<T>*> parameters, T learning_rate)
        : Optimizer<T>(parameters), learning_rate(learning_rate) {}
    void step() {
        for (auto& v : this->params) {
            v->raw() -= learning_rate * v->diff();
            v->clear();
        }
    }
};
template <typename T>
GradientDescent(std::initializer_list<AutoDiff<T>*> parameters, ...)
    -> GradientDescent<T>;

template <typename T> class Adam : public Optimizer<T> {
    struct AdamVariable {
        T m{0}, v{0};
    };
    std::map<AutoDiff<T>*, AdamVariable> adam_params;
    T learning_rate, beta1, beta2, epsilon;
    int t{0};

public:
    Adam(std::initializer_list<AutoDiff<T>*> parameters, T learning_rate, T beta1 = 0.9,
         T beta2 = 0.999, T epsilon = 1e-8)
        : Optimizer<T>(parameters), learning_rate(learning_rate), beta1(beta1),
          beta2(beta2), epsilon(epsilon) {
        for (auto& v : this->params) {
            adam_params[v];
        }
    }
    void step() {
        t++;
        for (auto& [v, adam_v] : adam_params) {
            auto& x = v->raw();
            auto g = v->diff();
            adam_v.m = beta1 * adam_v.m + (1 - beta1) * g;
            adam_v.v = beta2 * adam_v.v + (1 - beta2) * g * g;
            auto m_hat = adam_v.m / (1 - pow(beta1, t));
            auto v_hat = adam_v.v / (1 - pow(beta2, t));
            x -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            v->clear();
        }
    }
};

template <typename T>
Adam(std::initializer_list<AutoDiff<T>*> parameters, ...) -> Adam<T>;

}  // namespace optim