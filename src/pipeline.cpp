/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "pipeline.h"
#include "cusingler.cuh"
#include "io.h"
#include "time.h"
#include "timer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <thread>
using namespace std;

// bool Pipeline::score_data()
// {
//     //todo
// }



Pipeline::Pipeline()
{
}


bool Pipeline::finetune(int mod)
{
    if (mod == 0)
        cout << "rank by bin" << endl;
    else if (mod == 1)
    {
        cout << "rank by cnt" << endl;
    }
    else
    {
        cerr << "invalid mod." << endl;
        exit(-1);
    }
    cout << "work()" << endl;

    init();

    copyin(raw_data, raw_data.ctids, raw_data.ctidx, raw_data.ctdiff, raw_data.ctdidx, raw_data.ref, raw_data.qry);

    Timer timer("ms");
    auto  res = finetune(mod);
    cout << "finetune cost time(ms): " << timer.toc() << endl;

    // for (auto& c : res)
    //     cout<<raw_data.celltypes[c]<<endl;
    unordered_map< uint32, uint32 > m;

    // for (auto& [k,v] : m)
    //     cout<<k<<" "<<v<<endl;

    destroy();

    return true;
}

bool Pipeline::train(string filename, string ref_file, string qry_file)
{
    cout << "start training ref data." << endl;

    DataParser parser(ref_file, qry_file);
    parser.findIntersectionGenes();
    parser.loadRefData();
    parser.loadQryData();
    parser.preprocess();

    raw_data = parser.raw_data;
    readLabels(filename, raw_data);

    return true;
}

bool Pipeline::score()
{
    return true;
}