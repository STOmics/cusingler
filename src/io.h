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
    DataParser(int thread_num = 20);
    ~DataParser(){};

    // Load data from file
    bool prepareData(string ref_file, string qry_file); 

    bool trainData();
    bool findIntersectionGenes();
    bool loadRefData();
    bool loadQryData();
    bool preprocess();
    bool generateDenseMatrix(int step, uint64& max_uniq_gene);

    // Used for PyDataParser
    virtual bool prepareData(uint32 ref_height_, uint32 ref_width_,
        vector<float>& ref_data_, vector<int>& ref_indices_, vector<int>& ref_indptr_,
        uint32 qry_height_, uint32 qry_width_,
        vector<float>& qry_data_, vector<int>& qry_indices_, vector<int>& qry_indptr_,
        vector<string>& codes_, vector<int>& celltypes_,
        vector<string>& cellnames_, vector<string>& ref_geneidx_, vector<string>& qry_geneidx_) {}

private:
    bool loadRefMatrix(string filename);
    bool loadQryMatrix(string filename);

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

    vector<string> getGeneIndex(string filename, string gene_index);

    // For preprocess
    bool groupbyCelltypes();
    void resort();
    void removeCols(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                    map<uint32, uint32>& cols);

public:
    InputData raw_data;

    // Raw ref data from h5 file
    vector<float> ref_data;
    vector<int>   ref_indices;
    vector<int>   ref_indptr;
    uint32        ref_height, ref_width;
    vector<string> uniq_celltypes;
    vector<uint8> celltype_codes;
    vector<string> ref_genes;
    int           label_num;

    // Raw qry data from h5 file
    vector<float> qry_data;
    vector<int>   qry_indices;
    vector<int>   qry_indptr;
    uint32        qry_height, qry_width;
    vector<string> qry_cellnames;
    vector<string> qry_genes;

    set<uint32>    common_genes;
private:
    int    thread_num;

    // Train result of ref data
    vector<uint32> ref_train_idxs;
    vector<uint32> ref_train_values;
    set<uint32>    thre_genes;

    // Dense matrix
    vector<float> ref_dense;
    vector<float> qry_dense;


    // Filter genes in case there are differenet genes in ref and qry data
    bool                filter_genes;
    map<uint32, uint32> ref_gene_index;
    map<uint32, uint32> qry_gene_index;

    // Preprocess result
    vector<uint32> ref_ctidx;  // start index and len of each celltypes for ref cells
    vector<uint32> ref_ctids;
};

class PyDataParser : public DataParser
{
public:
    PyDataParser(int thread_num = 20);
    ~PyDataParser(){};

    // Load data from user input
    virtual bool prepareData(uint32 ref_height, uint32 ref_width,
        vector<float>& ref_data, vector<int>& ref_indices, vector<int>& ref_indptr,
        uint32 qry_height, uint32 qry_width,
        vector<float>& qry_data, vector<int>& qry_indices, vector<int>& qry_indptr,
        vector<string>& codes, vector<int>& celltypes,
        vector<string>& cellnames, vector<string>& ref_geneidx, vector<string>& qry_geneidx);

};