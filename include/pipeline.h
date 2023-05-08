/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include "types.h"

#include <set>
#include <string>
#include <vector>
using namespace std;

class Pipeline
{
public:
    Pipeline(string filename);
    ~Pipeline(){};
    bool preprocess();
    bool train(string ref_file, string qry_file);
    bool work(int mod);

private:
    // bool score_data();
    bool scale(vector< float >& src, const uint32 rows, const uint32 cols,
               vector< uint16 >& dest);
    bool filter_genes(vector< uint16 >& src, const uint32 rows, const uint32 cols,
                      set< uint32 >& genes);
    void filter();
    void resort();

private:
    InputData        rawdata;
    vector< uint32 > ctids;  // cell index of each cell type in ref data
    vector< uint32 > ctidx;

    vector< uint32 > ctdiff;  // gene index of between two cell types in ref data
    vector< uint32 > ctdidx;

    int label_num;

    vector< uint16 > ref;
    vector< uint16 > qry;
};