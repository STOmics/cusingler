/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */
 
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
using namespace pybind11::literals;

// void cusingler(string& ref_h5, string& qry_h5, string& stat_file, int cores, int gpuid)
// {
//     Pipeline pipeline = Pipeline(cores, gpuid);
//     pipeline.train(ref_h5, qry_h5);
//     pipeline.score();
//     pipeline.finetune();
//     pipeline.dump(stat_file);
// }

vector<vector<string>> run(int cores, int gpuid, uint32 ref_height, uint32 ref_width,
    vector<float>& ref_data, vector<int>& ref_indices, vector<int>& ref_indptr,
    uint32 qry_height, uint32 qry_width,
    vector<float>& qry_data, vector<int>& qry_indices, vector<int>& qry_indptr,
    vector<string>& codes, vector<int>& celltypes,
    vector<string>& cellnames, vector<string>& ref_geneidx, vector<string>& qry_geneidx,
    float quantile, float finetune_thre, int finetune_times
    )
{
    vector<vector<string>> res;
    
    PyPipeline pipeline = PyPipeline(cores, gpuid);
    pipeline.train(ref_height, ref_width, ref_data, ref_indices, ref_indptr,
        qry_height, qry_width, qry_data, qry_indices, qry_indptr,
        codes, celltypes, cellnames, ref_geneidx, qry_geneidx);
    pipeline.score(quantile/100, finetune_thre);
    pipeline.finetune(quantile/100, finetune_thre, finetune_times);
    res = pipeline.dump();

    return res;
}

PYBIND11_MODULE(cusingler, m) {
    m.doc() = R"pbdoc(
        cell annotation accelerated by GPU
        ----------------------------------

        .. currentmodule:: cusingler

        .. autosummary::
           :toctree: _generate

           cusingler
    )pbdoc";

    // m.def("cusingler", &cusingler, "cusingler function");
    m.def("run", &run, R"pbdoc(
        The main entry of cusingler, parameter:
        ---------------------------------------
        cores:: set cpu cores
        gpuid:: set gpu id, choose from [0,1,2...]
        ref_height/ref_width/ref_data/ref_indices/ref_indptr:: the attrs and data of reference matrix
        qry_height/qry_width/qry_data/qry_indices/qry_indptr:: the attrs and data of query matrix
        codes:: codes of celltype name, like ['abc', 'def', ...]
        celltype:: celltype name after decode, like [1,1,2,3,...]
        ref_gene_index:: gene names of reference file
        qry_gene_index:: gene names of query file
        quantile:: quantile will influence scoring and fine_tune result
        finetune_thre:: while in fine_tuning, if result greater than `max(result) - fine_tune_threshold`,
                        will be filtered.
        finetune_times:: default to 0, meaning that it will fine_tune until results decreasing to only 1;
                        If it is set to num(eg: 5), it will only loop only 5 times, and choose the first one
        )pbdoc",
        "cores"_a, "gpuid"_a, "ref_height"_a, "ref_width"_a, "ref_data"_a, "ref_indices"_a, "ref_indptr"_a,
        "qry_height"_a, "qry_width"_a, "qry_data"_a, "qry_indices"_a, "qry_indptr"_a,
        "codes"_a, "celltype"_a, "cellnames"_a, "ref_geneidx"_a, "qry_geneidx"_a,
        "quantile"_a=80, "finetune_thre"_a=0.05, "finetune_times"_a=0);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}