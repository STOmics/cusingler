/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#include "pipeline.h"
#include "io.h"
#include "cusingler.cuh"

#include <iostream>
using namespace std;

Pipeline::Pipeline(string filename)
{
    cout<<"start loading data."<<endl;
    if (!readInput(filename, rawdata))
        cerr<<"failed loading input h5 file."<<endl;
    else
        cout<<"success loading input h5 file."<<endl;
}

bool Pipeline::preprocess()
{
    cout<<"preprocess()"<<endl;

    // transfer the type of obs from string to int
    auto& celltypes = rawdata.celltypes;
    label_num = celltypes.size();

    unordered_map<string, uint8> ct2id;
    uint8 id = 0;
    for (auto& ct : celltypes)
        ct2id[ct] = id++;

    vector<vector<uint32>> aux_vec;
    aux_vec.resize(label_num);
    for (size_t i = 0; i < rawdata.obs_keys.size(); ++i)
    {
        // auto& k = rawdata.obs_keys[i];
        auto& v = rawdata.obs_values[i];
        if (ct2id.count(v))
            aux_vec[ct2id.at(v)].push_back(i);
    }
    uint32 start = 0;
    for (auto& vec : aux_vec)
    {
        ctidx.push_back(start);
        ctidx.push_back(vec.size());
        start += vec.size();
        ctids.insert(ctids.end(), vec.begin(), vec.end());
    }
    // for (size_t i = 0; i < ctidx.size()/2; ++i)
    //     cout<<"celltype's cells start: "<<ctidx[i*2]<<" len: "<<ctidx[i*2+1]<<endl;

    // re-construct trained data
    start = 0;
    for (int i = 0; i < label_num; ++i)
    {
        for (int j = 0; j < label_num; ++j)
        {
            if (i == j)
            {
                // padding zero
                ctdidx.push_back(0);
                ctdidx.push_back(0);
            }
            else
            {
                string key = celltypes[i]+'-'+celltypes[j];
                auto& vec = rawdata.trained[key];
                for (size_t k = 1; k < vec.size(); k+=2)
                    ctdiff.push_back(int(vec[k]));
                ctdidx.push_back(start);
                ctdidx.push_back(vec.size()/2);
                start += vec.size()/2;
            }
        }
    }
   
    // for (size_t i = 0; i < ctdidx.size()/2; ++i)
    //     cout<<"celltype's diff cells start: "<<ctdidx[i*2]<<" len: "<<ctdidx[i*2+1]<<endl;
        
    return true;
}

bool Pipeline::work()
{
    cout<<"work()"<<endl;

    init();

    copyin(rawdata, ctids, ctidx, ctdiff, ctdidx);

    auto res = finetune();
    unordered_map<uint32, uint32> m;
    for (size_t i = 0; i < res.size(); ++i)
    {
        // cout<<"cell idx: "<<i<<" celltype: "<<rawdata.celltypes[res[i]]<<endl;
        m[res[i]]++;
    }
    // for (auto& [k,v] : m)
    //     cout<<k<<" "<<v<<endl;

    destroy();

    return true;
}
