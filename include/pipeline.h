/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include "types.h"
#include "io.h"

#include <set>
#include <string>
#include <vector>
using namespace std;

class Pipeline
{
public:
    Pipeline(string ref_file, string qry_file, int rank_mode);
    ~Pipeline(){};

    bool train();
    bool score();
    bool finetune();
    
private:
    // Input parameters
    string ref_file;
    string qry_file;
    int rank_mode;

    // Manage input data
    DataParser *data_parser;
};