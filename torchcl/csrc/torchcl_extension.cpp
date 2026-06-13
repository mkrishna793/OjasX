#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

namespace py = pybind11;

static py::object py_alloc_func;
static py::object py_free_func;

static void cl_delete(void* ptr) {
    py::gil_scoped_acquire acquire;
    py_free_func(reinterpret_cast<uintptr_t>(ptr));
}

struct OpenCLAllocator : public c10::Allocator {
    c10::DataPtr allocate(size_t size) const override {
        py::gil_scoped_acquire acquire;
        py::object res = py_alloc_func(size);
        void* ptr = reinterpret_cast<void*>(res.cast<uintptr_t>());
        return {ptr, ptr, &cl_delete, c10::Device(c10::DeviceType::PrivateUse1, 0)};
    }
    c10::DeleterFnPtr raw_deleter() const override {
        return &cl_delete;
    }
};

static OpenCLAllocator g_cl_allocator;

void register_allocator(py::object alloc_func, py::object free_func) {
    py_alloc_func = alloc_func;
    py_free_func = free_func;
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_cl_allocator);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_allocator", &register_allocator, "Register python memory callbacks for OpenCL allocator");
}
