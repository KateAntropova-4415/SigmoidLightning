#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>

class Sigmoid {
public:
    static std::vector<double> sigmoid_vector(const std::vector<double>& input);
};

#endif // SIGMOID_H