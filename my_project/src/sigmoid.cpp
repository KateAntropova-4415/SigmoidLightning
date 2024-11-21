#include "sigmoid.h"
#include <cmath>

std::vector<double> Sigmoid::sigmoid_vector(const std::vector<double>& input) {
    std::vector<double> output;
    for (double value : input) {
        output.push_back(1.0 / (1.0 + std::exp(-value)));  // Функция сигмоида
    }
    return output;
}