/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include "types.h"

#include <vector>

bool init();

// for step: score data
bool        copyin_score(InputData& rawdata);
bool        destroy_score();
vector<int> get_label(InputData& rawdata, const uint64 max_uniq_gene);

// for step: fintune
bool copyin(InputData& rawdata, vector<uint32>& ctidx, vector<uint32>& ctdiff,
            vector<uint32>& ctdidx, vector<uint16>& ref, vector<uint16>& qry);
bool destroy();
std::vector<uint32> cufinetune(const uint64 max_uniq_gene);