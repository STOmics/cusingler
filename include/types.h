/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#pragma once

#include <unordered_map>
#include <string>
using namespace std;

struct InputData
{
    unordered_map<string, string> obs;  // cell name => cell type
};