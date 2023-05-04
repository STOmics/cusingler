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
    // unordered_map<string, string> obs;  // cell name => cell type
    vector< string > obs_keys, obs_values;

    vector< float > ref;  // expression matrix as reference, cell x gene
    uint32          ref_cell_num, ref_gene_num;

    vector< float > test;  // expression matrix as query data, cell x gene
    uint32          test_cell_num, test_gene_num;

    vector< float >  labels;     // initial annotated cell types, cell x celltype
    vector< string > celltypes;  // celltypes order

    unordered_map< string, vector< double > >
        trained;  // median expression difference for celltypes
};