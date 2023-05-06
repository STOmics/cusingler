/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include <string>
#include <set>
using namespace std;

#include "types.h"

bool readInput(string& filename, InputData& data);

class DataParser
{
public:
    DataParser(string ref_file, string qry_file);
    ~DataParser(){};

    bool loadRefData();
    bool loadQryData();

private:
    bool trainData();
    bool loadRefMatrix();

private:
    string ref_file;
    string qry_file;

    // Raw ref data from h5 file
    vector<float> ref_data;
    vector<int> ref_indices;
    vector<int> ref_indptr;
    uint32 ref_height, ref_width;
    vector<char*> uniq_celltypes;
    vector<uint8> celltype_codes;

    // Train result of ref data
    vector<uint32> ref_idxs;
    vector<uint32> ref_values;
    set<uint32> common_genes;
};