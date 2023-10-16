#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pipeline.h"
#include "types.h"

#include <string>
#include <vector>
using namespace std;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void cusingler(string& ref_h5, string& qry_h5, string& stat_file, int cores, int gpuid)
{
    Pipeline pipeline = Pipeline(cores, gpuid);
    pipeline.train(ref_h5, qry_h5);
    pipeline.score();
    pipeline.finetune();
    pipeline.dump(stat_file);
}

vector<vector<string>> run(int cores, int gpuid, uint32 ref_height, uint32 ref_width,
    vector<float>& ref_data, vector<int>& ref_indices, vector<int>& ref_indptr,
    uint32 qry_height, uint32 qry_width,
    vector<float>& qry_data, vector<int>& qry_indices, vector<int>& qry_indptr,
    vector<string>& codes, vector<int>& celltypes,
    vector<string>& cellnames, vector<string>& ref_geneidx, vector<string>& qry_geneidx
    )
{
    vector<vector<string>> res;
    
    PyPipeline pipeline = PyPipeline(cores, gpuid);
    pipeline.train(ref_height, ref_width, ref_data, ref_indices, ref_indptr,
        qry_height, qry_width, qry_data, qry_indices, qry_indptr,
        codes, celltypes, cellnames, ref_geneidx, qry_geneidx);
    pipeline.score();
    pipeline.finetune();
    res = pipeline.dump();

    return res;
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
    m.def("run", &run, "test function");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}