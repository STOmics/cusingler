/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#include "pipeline.h"
#include "io.h"
#include "timer.h"
#include "cusingler.cuh"
#include "time.h"
#include <iostream>
#include <thread>
#include <set>
#include <cmath>
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

bool Pipeline::filter_genes(vector<uint16>& src, const uint32 rows, const uint32 cols, set<uint32>& genes)
{
    vector<uint16> dest;
    dest.resize(size_t(rows) * genes.size(), 0);

    auto task = [&](size_t start, size_t rows, size_t cols){
        for (size_t i = 0; i < rows; ++i)
        {
            uint32 nj = 0;
            for (size_t j = 0; j < cols; ++j)
            {
                if (genes.count(cols-j-1) == 0)
                    continue;
                dest[(start+i)*genes.size()+nj] = src[(start+i)*cols+j];
                nj++;
            }
        }
    };
    constexpr int thread_num = 20;
    vector<thread> threads;
    size_t start = 0;
    size_t step = rows / thread_num;
    if (rows % thread_num != 0)
        step++;
    for (int i = 0; i < thread_num; ++i)
    {
        start = i*step;
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

    dest.swap(src);
    return true;
}

void Pipeline::filter()
{
    set<uint32> uniq_genes;
    int gene_thre = round(500 * pow((2/3.0), log2(2)));
    for (int i = 0; i < label_num; ++i)
    {
        for (int j = 0; j < label_num; ++j)
        {
            if (i == j)
                continue;
            int pos = ctdidx[(i * label_num + j) * 2];
            int len = ctdidx[(i * label_num + j) * 2 + 1];
            if (len > gene_thre)
                len = gene_thre;
            uniq_genes.insert(ctdiff.begin()+pos, ctdiff.begin()+pos+len);
            // cout<<"temp uniq genes size: "<<uniq_genes.size()<<endl;
        }
    }
    cout<<"useful genes: "<<uniq_genes.size()<<endl;
    unordered_map<uint32, uint32> gene_map;
    size_t idx = 0;
    for (auto v : uniq_genes)
        gene_map[v] = idx++;

    // re-construct trained data by filtering genes
    vector<uint32> _ctdiff;
    vector<uint32> _ctdidx;
    size_t start = 0;
    for (size_t i = 0; i < ctdidx.size(); i+=2)
    {
        auto s = ctdidx[i];
        auto l = ctdidx[i+1];
        if (s == 0 && l == 0)
        {
            _ctdidx.push_back(0);
            _ctdidx.push_back(0);
        }
        else
        {
            size_t ns = _ctdiff.size(), nl = 0;
            for (size_t j = s; j < s+l; j++)
            {
                if (uniq_genes.count(ctdiff[j]) == 0)
                    continue;
                _ctdiff.push_back(gene_map[ctdiff[j]]);
                nl++;
            }
            _ctdidx.push_back(ns);
            _ctdidx.push_back(nl);
        }
    }
    _ctdiff.swap(ctdiff);
    _ctdidx.swap(ctdidx);

    // for (int i = 0; i < rawdata.test_gene_num; ++i)
    //     if (uniq_genes.count(i) != 0 && qry[i] != 0)
    //         cout<<qry[i]<<" ";
    // cout<<endl;
    // filter genes for ref data
    filter_genes(ref, rawdata.ref_cell_num, rawdata.ref_gene_num, uniq_genes);

    // filter genes for qry data
    filter_genes(qry, rawdata.test_cell_num, rawdata.test_gene_num, uniq_genes);

    rawdata.ref_gene_num = uniq_genes.size();
    rawdata.test_gene_num = uniq_genes.size();

    // for (int i = 0; i < rawdata.test_gene_num; ++i)
    //     if (qry[i] != 0)
    //         cout<<i<<":"<<qry[i]<<" ";
    // cout<<endl;

    // int cnt = 0;
    // for (auto& v : uniq_genes)
    // {
    //     cout<<v<<" ";
    //     if (cnt++ >= 10)
    //         break;
    // }
    // cout<<endl;
}

void Pipeline::resort()
{
    vector<uint16> dest;
    dest.resize(ref.size(), 0);

    auto task = [&](size_t start){
        for (size_t i = start; i < label_num; i+=20)
        {
            size_t pos = ctidx[i*2];
            size_t len = ctidx[i*2+1];
            size_t idx = pos * rawdata.ref_gene_num;
            for (size_t j = pos; j < pos+len; ++j)
            {
                for (size_t k = 0; k < rawdata.ref_gene_num; ++k)
                    dest[idx++] = ref[ctids[j]*rawdata.ref_gene_num+k];
            }
        }
    };
    constexpr int thread_num = 20;
    vector<thread> threads;
    for (int i = 0; i < thread_num; ++i)
    {
        thread th(task, i);
        threads.push_back(std::move(th));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    dest.swap(ref);
    for (uint32 i = 0; i < ctids.size(); ++i)
        ctids[i] = i;

}

bool Pipeline::score_data(int mod)
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


    copyin(rawdata, ctids, ctidx, ctdiff, ctdidx, ref, qry);
    
    cout<<"score_data"<<endl;
    //get score
    get_label(rawdata,mod);

    //compare with threshold
    //todo
   
    getchar();
    //flip


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
        // cout<<start-vec.size()<<" "<<vec.size()<<" "<<ctids.size()<<endl;
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

    // filter genes of datas
    timer.tic();
    filter();
    cout<<"filter genes cost time(ms): "<<timer.toc()<<endl;
    cout<<"new qry size: "<< rawdata.test_cell_num << " x "<<rawdata.test_gene_num<<endl;
    cout<<"new ref size: "<< rawdata.ref_cell_num << " x "<<rawdata.ref_gene_num<<endl;

    // re-sort ref data groupby celltype
    resort();
    cout<<"re-sort ref by celltype cost time(ms): "<<timer.toc()<<endl;

    // exit(0);

    return true;
}

bool Pipeline::work(int mod)
{
    if(mod == 0)
        cout<<"rank by bin"<<endl;
    else if (mod==1)
    {
        cout<<"rank by cnt"<<endl;
    }
    else
    {
        cerr<<"invalid mod."<<endl;
        exit(-1);
    }
    cout<<"work()"<<endl;

    init();


    copyin(rawdata, ctids, ctidx, ctdiff, ctdidx, ref, qry);

    Timer timer("ms");
    auto res = finetune(mod);
    cout<<"finetune cost time(ms): "<<timer.toc()<<endl;

    unordered_map<uint32, uint32> m;

    // for (auto& [k,v] : m)
    //     cout<<k<<" "<<v<<endl;

    destroy();

    return true;
}
