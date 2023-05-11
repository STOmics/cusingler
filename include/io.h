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

bool readLabels(string filename, InputData& data);

class DataParser
{
public:
    DataParser(string ref_file, string qry_file, int thread_num=20);
    ~DataParser(){};

    bool findIntersectionGenes();
    bool loadRefData();
    bool loadQryData();
    bool preprocess();

private:
    bool trainData();
    bool loadRefMatrix();
    bool csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices, int width, vector<float>& res);
    bool csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices, set<uint32>& cols, vector<float>& res);

    bool loadQryMatrix();

    vector<char*> getGeneIndex(string filename, string gene_index);

    // For preprocess
    bool groupbyCelltypes();
    bool scale(vector< float >& src, const uint32 rows, const uint32 cols,
               vector< uint16 >& dest);
    bool filterGenes(vector< uint16 >& src, const uint32 rows, const uint32 cols,
                      set< uint32 >& genes);
    void filter();
    void resort();

public:
    InputData raw_data;

private:
    string ref_file;
    string qry_file;
    int thread_num;

    // Raw ref data from h5 file
    vector<float> ref_data;
    vector<int> ref_indices;
    vector<int> ref_indptr;
    uint32 ref_height, ref_width;
    vector<char*> uniq_celltypes;
    int label_num;
    vector<uint8> celltype_codes;

    // Train result of ref data
    vector<uint32> ref_train_idxs;
    vector<uint32> ref_train_values;
    set<uint32> common_genes;

    // Dense matrix
    vector<float> ref_dense;
    vector<float> qry_dense;

    // Raw qry data from h5 file
    vector<float> qry_data;
    vector<int> qry_indices;
    vector<int> qry_indptr;
    uint32 qry_height, qry_width;
    vector<char*> genes;

    // Filter genes in case there are differenet genes in ref and qry data
    bool filter_genes;
    set<uint32> ref_gene_index;
    set<uint32> qry_gene_index;

    // Preprocess result
    vector< uint32 > ref_ctids;  // cell index of each cell type in ref data
    vector< uint32 > ref_ctidx;
};