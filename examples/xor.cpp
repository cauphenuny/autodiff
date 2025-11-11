#include "optim.hpp"
#include "variable.hpp"

#include <cmath>
#include <random>
#include <utility>
#include <vector>

class Layer {
    std::vector<std::vector<var>> weights;

public:
    Layer(int input_size, int output_size) {
        weights.resize(output_size, std::vector<var>(input_size));
        // Xavier/Glorot uniform initialization
        static thread_local std::mt19937 gen(std::random_device{}());
        double limit = std::sqrt(
            6.0 / (static_cast<double>(input_size) + static_cast<double>(output_size)));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] = var(dist(gen));
            }
        }
    }
    std::vector<var> forward(const std::vector<var>& input) {
        std::vector<var> output;
        for (const auto& w_row : weights) {
            var sum = 0;
            for (size_t j = 0; j < w_row.size(); j++) {
                sum = sum + w_row[j] * input[j];
            }
            output.push_back(sum);
        }
        return output;
    }
    auto parameters() {
        std::vector<AutoDiff<double>*> params;
        for (auto& w_row : weights) {
            for (auto& w : w_row) {
                params.push_back(&w);
            }
        }
        return params;
    }
};

class XORModel {
    Layer layer1;
    Layer layer2;

public:
    XORModel(int hidden_size = 8) : layer1(2, hidden_size), layer2(hidden_size, 1) {}
    auto parameters() {
        auto params1 = layer1.parameters();
        auto params2 = layer2.parameters();
        params1.insert(params1.end(), params2.begin(), params2.end());
        return params1;
    }
    var forward(var x1, var x2) {
        auto hidden = layer1.forward({x1, x2});
        std::vector<var> activated;
        for (auto& h : hidden) {
            activated.push_back(1 / (1 + exp(-h)));  // Sigmoid activation
        }
        auto output = layer2.forward(activated);
        auto result = 1 / (1 + exp(-output[0]));  // Sigmoid activation
        return result;
    }

    void fit(int max_epoch) {
        optim::Adam<double> optimizer(this->parameters(), 0.1);
        for (int epoch = 0; epoch < max_epoch; epoch++) {
            std::vector<std::pair<std::pair<int, int>, int>> data = {
                {{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 1}, {{1, 1}, 0}};
            for (auto [x, y] : data) {
                var output = this->forward(x.first, x.second);
                var loss = -(y * log(output) + (1 - y) * log(1 - output));
                loss.propagate();
                optimizer.step();
            }
        }
    }
};

int main() {
    XORModel model;
    model.fit(1000);
    for (int x1 : {0, 1}) {
        for (int x2 : {0, 1}) {
            var output = model.forward(x1, x2);
            std::cout << "XOR(" << x1 << ", " << x2 << ") = " << output.raw() << std::endl;
        }
    }
    return 0;
}