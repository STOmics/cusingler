#include <pybind11/pybind11.h>

#include "pipeline.h"

#include <string>
using namespace std;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void cusingler(string& ref_h5, string& qry_h5, string& stat_file, int cores, int gpuid)
{
    Pipeline pipeline = Pipeline(ref_h5, qry_h5, stat_file, cores, gpuid);
    pipeline.train();
    pipeline.score();
    pipeline.finetune();
    pipeline.dump();
}

PYBIND11_MODULE(cusingler, m) {
    m.doc() = R"pbdoc(
        cell annotation accelerated by GPU
        -----------------------

        .. currentmodule:: cusingler

        .. autosummary::
           :toctree: _generate

           cusingler
    )pbdoc";

    m.def("cusingler", &cusingler, "cusingler function");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}