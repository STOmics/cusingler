/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#include "pipeline.h"
#include "io.h"
#include "timer.h"
#include "cusingler.cuh"

#include <iostream>
#include <thread>
#include <set>
#include <cassert>
using namespace std;

bool Pipeline::scale(vector<float>& src, const uint32 rows, const uint32 cols, vector<uint16>& dest)
{
    dest.resize(src.size(), 0);

    std::mutex m;
    auto task = [&](size_t start, size_t rows, size_t cols){
        size_t max_index = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            set<float> uniq;
            for (size_t j = 0; j < cols; ++j)
                uniq.insert(src[start+i*cols+j]);
            assert(uniq.size() < 65536);

            vector<float> order(uniq.begin(), uniq.end());
            unordered_map<float, uint16> index;
            for (uint16 j = 0; j < order.size(); ++j)
            {
                index[order[j]] = j;
            }
            for (size_t j = 0; j < cols; ++j)
                dest[start+i*cols+j] = index[src[start+i*cols+j]];
            max_index = max(max_index, index.size());
        }

        // std::lock_guard<std::mutex> lg(m);
        // cout<<"max index: "<<max_index<<endl;
    };
    constexpr int thread_num = 20;
    vector<thread> threads;
    size_t start = 0;
    size_t step = rows / thread_num;
    if (rows % thread_num != 0)
        step++;
    for (int i = 0; i < thread_num; ++i)
    {
        start = i*step*cols;
        if (i == (thread_num-1))
            step = rows - i * step;
        thread th(task, start, step, cols);
        // cout<<start<<" "<<step<<" "<<cols<<endl;
        threads.push_back(std::move(th));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    return true;
}

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
        
    // unordered_map<float, size_t> bincount;
    // for (auto& f : rawdata.test)
    //     bincount[f]++;
    // cout<<"test len: "<<rawdata.test.size()<<" uniq len: "<<bincount.size()<<endl;

    Timer timer("ms");
    scale(rawdata.ref, rawdata.ref_cell_num, rawdata.ref_gene_num, ref);
    cout<<"scale ref cost time(ms): "<<timer.toc()<<endl;
    vector<float>().swap(rawdata.ref);

    timer.tic();
    scale(rawdata.test, rawdata.test_cell_num, rawdata.test_gene_num, qry);
    cout<<"scale qry cost time(ms): "<<timer.toc()<<endl;
    vector<float>().swap(rawdata.test);

    // exit(0);

    return true;
}

bool Pipeline::work()
{
    cout<<"work()"<<endl;

    init();

    copyin(rawdata, ctids, ctidx, ctdiff, ctdidx, ref, qry);

    Timer timer("ms");
    auto res = finetune();
    cout<<"finetune cost time(ms): "<<timer.toc()<<endl;

    unordered_map<uint32, uint32> m;
    for (size_t i = 0; i < res.size(); ++i)
    {
        // cout<<"cell idx: "<<i<<" celltype: "<<rawdata.celltypes[res[i]]<<endl;
        m[res[i]]++;
    }
    for (auto& [k,v] : m)
        cout<<k<<" "<<v<<endl;

    destroy();

    return true;
}
