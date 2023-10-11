/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#pragma once

#include <map>
#include <set>
#include <string>
using namespace std;

#include "types.h"

class DataParser
{
public:
    DataParser(string ref_file, string qry_file, int thread_num = 20);
    ~DataParser(){};

    bool trainData();

    bool findIntersectionGenes();
    bool loadRefData();
    bool loadQryData();
    bool preprocess();
    bool generateDenseMatrix(int step, uint64& max_uniq_gene);

private:
    bool loadRefMatrix();
    bool loadQryMatrix();

    // Transform csr to dense matrix
    bool csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                   int width, vector<float>& res);
    // Transform csr to dense matrix with column filtered
    bool csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                   set<uint32>& cols, vector<uint16>& res, uint64& max_uniq_gene);

    // Filter csr data by columns
    bool csrFilter(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                   set<uint32>& cols, vector<uint16>& res_data, vector<int>& ref_indptr,
                   vector<int>& ref_indices, uint64& max_uniq_gene);

    vector<char*> getGeneIndex(string filename, string gene_index);

    // For preprocess
    bool groupbyCelltypes();
    void resort();
    void removeCols(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                    map<uint32, uint32>& cols);

public:
    InputData raw_data;

private:
    string ref_file;
    string qry_file;
    int    thread_num;

    // Raw ref data from h5 file
    vector<float> ref_data;
    vector<int>   ref_indices;
    vector<int>   ref_indptr;
    uint32        ref_height, ref_width;
    vector<char*> uniq_celltypes;
    int           label_num;
    vector<uint8> celltype_codes;

    // Train result of ref data
    vector<uint32> ref_train_idxs;
    vector<uint32> ref_train_values;
    set<uint32>    common_genes;
    set<uint32>    thre_genes;

    // Dense matrix
    vector<float> ref_dense;
    vector<float> qry_dense;

    // Raw qry data from h5 file
    vector<float> qry_data;
    vector<int>   qry_indices;
    vector<int>   qry_indptr;
    uint32        qry_height, qry_width;
    vector<char*> genes;
    vector<char*> qry_cellnames;

    // Filter genes in case there are differenet genes in ref and qry data
    bool                filter_genes;
    map<uint32, uint32> ref_gene_index;
    map<uint32, uint32> qry_gene_index;

    // Preprocess result
    vector<uint32> ref_ctidx;  // start index and len of each celltypes for ref cells
    vector<uint32> ref_ctids;
};