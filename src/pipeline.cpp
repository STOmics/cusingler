/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "pipeline.h"
#include "cusingler.cuh"
#include "timer.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>
using namespace std;

Pipeline::Pipeline(int cores, int gpuid)
    : cores(cores), gpuid(gpuid)
{
}

bool Pipeline::train(string ref_file, string qry_file)
{
    if (!initGPU(gpuid))
    {
        cerr << "Fail to initlize gpu with id: " << gpuid << endl;
        exit(-1);
    }

    // cout << "start training ref data." << endl;
    Timer timer;

    data_parser = new DataParser(cores);
    data_parser->prepareData(ref_file, qry_file);
    data_parser->findIntersectionGenes();
    data_parser->loadRefData();
    data_parser->trainData();
    data_parser->loadQryData();
    data_parser->preprocess();
    cout << "train data cost time(s): " << timer.toc() << endl;

    return true;
}

bool Pipeline::score(float quantile, float finetune_thre)
{
    // cout << "start get labels." << endl;

    Timer timer;

    uint64 max_uniq_gene = 0;
    data_parser->generateDenseMatrix(0, max_uniq_gene);
    // cout << "max uniq gene: " << max_uniq_gene << endl;
    auto& raw_data = data_parser->raw_data;
    raw_data.labels.clear();
    raw_data.labels.resize(raw_data.qry_height * raw_data.ct_num, 0);

    cells = raw_data.qry_cellnames;

    if (!copyin_score(raw_data))
    {
        exit(-1);
    }

    auto first_label_index = get_label(raw_data, max_uniq_gene, cores, quantile, finetune_thre);
    for (auto& i : first_label_index)
        first_labels.push_back(raw_data.celltypes[i]);

    destroy_score();

#ifdef DEBUG
    ofstream ofs("temp_score.tsv");
    for (uint32 i = 0; i < first_labels.size(); ++i)
    {
        for (uint32 j = 0; j < raw_data.ct_num; ++j)
            ofs << raw_data.labels[i*raw_data.ct_num+j] << "\t";
        ofs<<endl;
    }
    ofs.close();
#endif

    cout << "score data cost time(s): " << timer.toc() << endl;
    return true;
}

bool Pipeline::finetune(float quantile, float finetune_thre, int finetune_times)
{
    // cout << "start finetune." << endl;
    Timer timer("s");

    uint64 max_uniq_gene = 0;
    data_parser->generateDenseMatrix(1, max_uniq_gene);
    // cout << "max uniq gene: " << max_uniq_gene << endl;

    auto& raw_data = data_parser->raw_data;

    if (!copyin(raw_data))
    {
        exit(-1);
    }
    auto res = cufinetune(max_uniq_gene, quantile, finetune_thre, finetune_times);
    for (auto& c : res)
        final_labels.push_back(raw_data.celltypes[c]);

    destroy();
    cout << "finetune cost time(s): " << timer.toc() << endl;

    return true;
}

bool Pipeline::dump(string stat_file)
{
    ofstream ofs(stat_file);
    ofs << "cell\tfirstLabel\tfinalLabel\n";
    for (uint32 i = 0; i < final_labels.size(); ++i)
        ofs << cells[i] << "\t" << first_labels[i] << "\t" << final_labels[i] << endl;
    ofs.close();
    return true;
}

PyPipeline::PyPipeline(int cores, int gpuid)
    : Pipeline(cores, gpuid)
{
}

bool PyPipeline::train(uint32 ref_height, uint32 ref_width,
    vector<float>& ref_data, vector<int>& ref_indices, vector<int>& ref_indptr,
    uint32 qry_height, uint32 qry_width,
    vector<float>& qry_data, vector<int>& qry_indices, vector<int>& qry_indptr,
    vector<string>& codes, vector<int>& celltypes,
    vector<string>& cellnames, vector<string>& ref_geneidx, vector<string>& qry_geneidx)
{
    if (!initGPU(gpuid))
    {
        cerr << "Fail to initlize gpu with id: " << gpuid << endl;
        exit(-1);
    }

    // cout << "start training ref data." << endl;
    Timer timer;

    data_parser = new PyDataParser(cores);
    data_parser->prepareData(ref_height, ref_width, ref_data, ref_indices, ref_indptr,
        qry_height, qry_width, qry_data, qry_indices, qry_indptr,
        codes, celltypes, cellnames, ref_geneidx, qry_geneidx);
    data_parser->findIntersectionGenes();
    data_parser->loadRefData();
    data_parser->trainData();
    data_parser->loadQryData();
    data_parser->preprocess();
    cout << "train data cost time(s): " << timer.toc() << endl;

    return true;
}

vector<vector<string>> PyPipeline::dump()
{
    vector<vector<string>> res;
    for (uint32 i = 0; i < final_labels.size(); ++i)
        res.push_back({cells[i], first_labels[i], final_labels[i]});
    return res;
}
