/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#pragma once

#include <unordered_map>
#include <string>
#include <vector>
using namespace std;

struct InputData
{
    unordered_map<string, string> obs;  // cell name => cell type
    vector<float> ref; // expression matrix as reference, cell x gene
    vector<float> test; // expression matrix as query data, cell x gene
    vector<float> celltype; // initial annotated cell types, cell x celltype
    unordered_map<string, vector<double>> trained; // median expression difference for celltypes
};