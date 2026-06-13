#include <torch/extension.h>
int add(int i, int j) { return i + j; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add, "A function that adds two numbers"); }
