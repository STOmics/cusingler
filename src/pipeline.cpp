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
#include <iostream>
#include <set>
#include <thread>
using namespace std;

Pipeline::Pipeline(string ref_file, string qry_file, int rank_mode) :
    ref_file(ref_file), qry_file(qry_file), rank_mode(rank_mode)
{
}

bool Pipeline::train()
{
    cout << "start training ref data." << endl;

    data_parser = new DataParser(ref_file, qry_file);
    data_parser->findIntersectionGenes();
    data_parser->loadRefData();
    data_parser->trainData();
    data_parser->loadQryData();
    data_parser->preprocess();

    return true;
}

bool Pipeline::score()
{
    cout << "start get labels." << endl;

    data_parser->generateDenseMatrix(0);
    auto& raw_data = data_parser->raw_data;
    raw_data.labels.resize(raw_data.ct_num * raw_data.qry_height, 0);    
    // for (int i = 0; i < 34; ++i)
    //     cout<<raw_data.labels[i]<<" ";
    // cout<<endl;

    // init();
    // copyin_score(raw_data);
    
    //get score
    Timer timer("ms");

    // get_label(raw_data, rank_mode);

    cout << "score cost time(ms): " << timer.toc() << endl;
    // for (int i = 0; i < 34; ++i)
    //     cout<<raw_data.labels[i]<<" ";
    // cout<<endl;
    
    // destroy_score();

    return true;
}


bool Pipeline::finetune()
{
    cout << "start finetune." << endl;

    data_parser->generateDenseMatrix(1);

    auto& raw_data = data_parser->raw_data;

    init();  //init and copy  in scoredata

    copyin(raw_data, raw_data.ctidx, raw_data.ctdiff, raw_data.ctdidx, raw_data.ref, raw_data.qry);

    Timer timer("ms");
    auto  res = cufinetune(rank_mode);
    cout << "finetune cost time(ms): " << timer.toc() << endl;

    for (auto& c : res)
        cout<<raw_data.celltypes[c]<<endl;

    destroy();

    return true;
}

