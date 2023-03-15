
#include "cusingler.cuh"

#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <thread>
#include <iostream>

#include "cuda_runtime.h"


float* d_ref, *d_qry, *d_labels;
uint32 ref_height, ref_width, qry_height, qry_width;
uint32 ct_num;
uint32* d_ctids, *d_ctidx, *d_ctdiff, *d_ctdidx;

// unit is MB
uint32 getUsedMem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (total-free)/1024/1024;
}

bool init()
{
    d_ref = NULL;
    d_qry = NULL;
    d_labels = NULL;
    ref_height = ref_width = qry_height = qry_width = 0;
    ct_num = 0;
    d_ctids = NULL;
    d_ctidx = NULL;
    d_ctdiff = NULL;
    d_ctdidx = NULL;

    return true;
}

bool destroy()
{
    cudaFree(d_ref);
    cudaFree(d_qry);
    cudaFree(d_labels);
    cudaFree(d_ctids);
    cudaFree(d_ctidx);
    cudaFree(d_ctdiff);
    cudaFree(d_ctdidx);

    return true;
}

bool copyin(InputData& rawdata, vector<uint32>& ctids, vector<uint32>& ctidx, vector<uint32>& ctdiff, vector<uint32>& ctdidx)
{
    ref_height = rawdata.ref_gene_num;
    ref_width = rawdata.ref_cell_num;
    qry_height = rawdata.test_gene_num;
    qry_width = rawdata.test_cell_num;
    ct_num = rawdata.celltypes.size();

    cudaMalloc((void**)&d_ref, ref_height * ref_width * sizeof(float));
    cudaMalloc((void**)&d_qry, qry_height * qry_width * sizeof(float));
    cudaMalloc((void**)&d_labels, qry_height * ct_num * sizeof(float));
    cudaMalloc((void**)&d_ctids, ctids.size() * sizeof(uint32));
    cudaMalloc((void**)&d_ctidx, ctidx.size() * sizeof(uint32));
    cudaMalloc((void**)&d_ctdiff, ctdiff.size() * sizeof(uint32));
    cudaMalloc((void**)&d_ctdidx, ctdidx.size() * sizeof(uint32));

    cudaMemcpy(d_ref, rawdata.ref.data(), ref_height * ref_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qry, rawdata.test.data(), qry_height * qry_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, rawdata.labels.data(), qry_height * ct_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctidx, ctidx.data(), ctidx.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctdiff, ctdiff.data(), ctdiff.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctdidx, ctdidx.data(), ctdidx.size() * sizeof(uint32), cudaMemcpyHostToDevice);

    // std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout<<"used gpu mem(MB): "<<getUsedMem()<<std::endl;

    return true;
}

bool finetune_round(float* qry, float* labels)
{
    // TODO
    return true;
}

bool finetune()
{
    // process each cell
    for (int i = 0; i < qry_height; ++i)
    {
        finetune_round(d_qry+i*qry_width, d_labels+i*ct_num);
    }
 
    return true;
}