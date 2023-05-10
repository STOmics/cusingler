/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include "types.h"

#include <vector>

bool init();
bool destroy();
bool copyin(InputData& rawdata, vector<uint32>& ctids, vector<uint32>& ctidx, vector<uint32>& ctdiff, vector<uint32>& ctdidx,
    vector<uint16>& ref, vector<uint16>& qry);
std::vector<uint32> cufinetune(int mod);
std::vector<uint32> get_label(InputData& rawdata,int mod);