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



bool Pipeline::score(int mod)
{
    if(mod == 0)
        cout<<"score_data by bin"<<endl;
    else if (mod==1)
    {
        cout<<"score_data by cnt"<<endl;
    }
    else
    {
        cerr<<"invalid mod."<<endl;
        exit(-1);
    }

    init();
    copyin_score(raw_data);
    
    //get score
    Timer timer("ms");

    auto py_labels = raw_data.labels;

    get_label(raw_data, mod);
    cout << "get_label cost time(ms): " << timer.toc() << endl;

    size_t res = 0;
    for (int i = 0; i < raw_data.labels.size(); ++i)
    {
        // if (i < 34)
        // {
        //     cout<<py_labels[i]<<" "<<raw_data.labels[i] <<endl;
        // }
        if (raw_data.labels[i] != py_labels[i])
        {
            cout<<i<<" "<<py_labels[i]<<" "<<raw_data.labels[i] <<" "<< i / raw_data.ct_num<<endl;
            res++;
        }
    }
    cout<<"labels size: "<<py_labels.size()<<" diff size: "<<res<<endl;
    destroy_score();

    return true;
}

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

   // init();  //init and copy  in scoredata

    copyin(raw_data, raw_data.ctids, raw_data.ctidx, raw_data.ctdiff, raw_data.ctdidx, raw_data.ref, raw_data.qry);

    Timer timer("ms");
    auto  res = cufinetune(mod);
    cout << "finetune cost time(ms): " << timer.toc() << endl;

    for (auto& c : res)
        cout<<raw_data.celltypes[c]<<endl;
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