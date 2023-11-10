/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include "io.h"
#include "types.h"

#include <set>
#include <string>
#include <vector>
using namespace std;

class Pipeline
{
public:
    Pipeline(int cores, int gpuid);
    ~Pipeline(){};

    bool train(string ref_file, string qry_file);
    bool score(float quantile, float finetune_thre);
    bool finetune(float quantile, float finetune_thre, int finetune_times);
    bool dump(string stat_file);

public:
    // Input parameters
    int    cores;
    int    gpuid;

    // Manage input data
    DataParser* data_parser;

    // Stat data
    vector<string>  cells;
    vector<string> first_labels;
    vector<string> final_labels;
};

class PyPipeline : public Pipeline
{
public:
    PyPipeline(int cores, int gpuid);
    ~PyPipeline(){};

    bool train(uint32 ref_height, uint32 ref_width,
        vector<float>& ref_data, vector<int>& ref_indices, vector<int>& ref_indptr,
        uint32 qry_height, uint32 qry_width,
        vector<float>& qry_data, vector<int>& qry_indices, vector<int>& qry_indptr,
        vector<string>& codes, vector<int>& celltypes,
        vector<string>& cellnames, vector<string>& ref_geneidx, vector<string>& qry_geneidx);
    vector<vector<string>> dump();

// public:
//     PyDataParser* data_parser;

};