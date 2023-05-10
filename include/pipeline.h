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
    Pipeline();
    ~Pipeline(){};

    bool train(string filename, string ref_file, string qry_file);
    bool score();
    bool finetune(int mod);

// private:
    // bool score_data();
    

private:
    InputData        raw_data;

};