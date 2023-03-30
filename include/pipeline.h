/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#pragma once

#include "types.h"

#include <string>
#include <vector>
using namespace std;

class Pipeline
{
public:
    Pipeline(string filename);
    ~Pipeline() {};
    bool preprocess();
    bool work();

private:
    InputData rawdata;
    vector<uint32> ctids;   // cell index of each cell type in ref data
    vector<uint32> ctidx;

    vector<uint32> ctdiff;  // gene index of between two cell types in ref data
    vector<uint32> ctdidx;

    int label_num;
};