/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

typedef unsigned char  uint8;
typedef unsigned short uint16;
typedef unsigned int   uint32;
typedef unsigned long  uint64;

struct InputData
{
    vector<uint16> ref;  // expression matrix as reference, cell x gene
    vector<uint16> ref_data;
    vector<int> ref_indptr, ref_indices;
    uint32         ref_height, ref_width;

    vector<uint16> qry;  // expression matrix as query data, cell x gene
    vector<uint16> qry_data;
    vector<int> qry_indptr, qry_indices;
    uint32         qry_height, qry_width;
    vector<char*>  qry_cellnames;

    vector<float>  labels;     // initial annotated cell types, cell x celltype
    vector<string> celltypes;  // celltypes order
    int            ct_num;     // count of unique celltypes

    vector<uint32> ctdiff;  // gene index of between two cell types in ref data
    vector<uint32> ctdidx;

    vector<uint32> ctidx;
};

struct Slice
{
    bool on;

    uint32 qry_rows;
    uint32 ref_rows;

    uint32 qry_steps;
    uint32 ref_steps;
};