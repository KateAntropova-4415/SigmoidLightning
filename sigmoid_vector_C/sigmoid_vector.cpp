#include <cmath>
#include <vector>
#include <iostream>

extern "C" {
    void sigmoid_vector(const double* input, double* output, int size) {
        for (int i = 0; i < size; ++i) {
            output[i] = 1.0 / (1.0 + std::exp(-input[i]));
        }
    }
}
