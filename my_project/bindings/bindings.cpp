#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sigmoid.h"

namespace py = pybind11;

PYBIND11_MODULE(sigmoid_core, m) {
    m.doc() = "Sigmoid function implemented in C++";

    py::class_<Sigmoid>(m, "Sigmoid")
        .def_static("sigmoid_vector", &Sigmoid::sigmoid_vector, "Compute sigmoid for a vector of doubles");
}
