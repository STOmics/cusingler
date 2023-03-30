
#include "cusingler.cuh"

#include <stdlib.h>
#include <stdio.h>

#include <cmath>
#include <chrono>
#include <thread>
#include <iostream>
#include <set>
#include <algorithm>

#include "cuda_runtime.h"
#include "math_constants.h"

cudaError_t errcode;
cudaStream_t stream;
float* d_ref, *d_qry;
vector<float> h_labels;
uint32 ref_height, ref_width, qry_height, qry_width;
uint32 ct_num;
uint32* d_ctids;
vector<uint32> h_ctidx;
vector<uint32> h_ctdiff, h_ctdidx;
size_t pitchref;
size_t pitchqry;

uint32* d_gene_idx;
float* d_qry_line, *d_qry_rank;
float* d_ref_lines, *d_ref_rank;
float *d_score;

// unit is MB
uint32 getUsedMem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (total-free)/1024/1024;
}
bool err_check()
{
    if(errcode!=cudaSuccess)
        std::cout << "cudaerrcode:"<<errcode<<" line = %d" << __LINE__<<endl;
     return true;
}
bool init()
{
    stream =NULL;
    d_ref = NULL;
    d_qry = NULL;
    ref_height = ref_width = qry_height = qry_width = 0;
    ct_num = 0;
    d_ctids = NULL;
    pitchref=0;
    pitchqry=0;
    return true;
}

bool destroy()
{
    cudaFree(d_ref);
    cudaFree(d_qry);
    // cudaFree(d_labels);
    cudaFree(d_ctids);
    cudaStreamDestroy(stream);
    // cudaFree(d_ctidx);
    // cudaFree(d_ctdiff);
    // cudaFree(d_ctdidx);

    cudaFree(d_gene_idx);
    cudaFree(d_qry_line);
    cudaFree(d_qry_rank);
    cudaFree(d_ref_lines);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    return true;
}

bool copyin(InputData& rawdata, vector<uint32>& ctids, vector<uint32>& ctidx, vector<uint32>& ctdiff, vector<uint32>& ctdidx)
{
    

    ref_height = rawdata.ref_cell_num;
    ref_width = rawdata.ref_gene_num;
    qry_height = rawdata.test_cell_num;
    qry_width = rawdata.test_gene_num;
    ct_num = rawdata.celltypes.size();

    // float max_val = 0;
    // for (int i = 0; i < qry_width; ++i)
    // {
    //     max_val = max(max_val, rawdata.test[i]);
    // }
    // cout<<"qry width: "<<qry_width<<endl;
    // cout<<"qry max value: "<<max_val<<endl;
    //

    cudaError_t cudaerr;
    cudaerr=cudaMallocPitch((void**)&d_ref,&pitchref,ref_width*sizeof(float),ref_height);
    cudaMallocPitch((void**)&d_qry,&pitchqry,qry_width*sizeof(float),qry_height);
    
    std::cout<<"pitchref: "<<pitchref<<std::endl;
    std::cout<<"pitchqry: "<<pitchqry<<std::endl;

    //cudaMalloc((void**)&d_ref, ref_height * ref_width * sizeof(float));
    //cudaMalloc((void**)&d_qry, qry_height * qry_width * sizeof(float));
    // cudaMalloc((void**)&d_labels, qry_height * ct_num * sizeof(float));
    cudaMalloc((void**)&d_ctids, ctids.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctidx, ctidx.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctdiff, ctdiff.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctdidx, ctdidx.size() * sizeof(uint32));

    cudaMemcpy2DAsync(d_ref,pitchref, rawdata.ref.data(), ref_width * sizeof(float),ref_width * sizeof(float),ref_height,cudaMemcpyHostToDevice,stream);
    cudaMemcpy2DAsync(d_qry,pitchqry, rawdata.test.data(),qry_width * sizeof(float),qry_width * sizeof(float),qry_height,cudaMemcpyHostToDevice,stream);
   

    // cudaMemcpyAsync(d_ref, rawdata.ref.data(), ref_height * ref_width * sizeof(float), cudaMemcpyHostToDevice,stream);
    // cudaMemcpyAsync(d_qry, rawdata.test.data(), qry_height * qry_width * sizeof(float), cudaMemcpyHostToDevice,stream);
    // // cudaMemcpy(d_labels, rawdata.labels.data(), qry_height * ct_num * sizeof(float), cudaMemcpyHostToDevice);
    h_labels = rawdata.labels;
    cudaMemcpyAsync(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice,stream);
    // cudaMemcpy(d_ctidx, ctidx.data(), ctidx.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    h_ctidx = ctidx;
    // cudaMemcpy(d_ctdiff, ctdiff.data(), ctdiff.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_ctdidx, ctdidx.data(), ctdidx.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    h_ctdiff = ctdiff;
    h_ctdidx = ctdidx;
    cudaStreamSynchronize(stream);
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout<<"used gpu mem(MB): "<<getUsedMem()<<std::endl;

    cudaMalloc((void**)&d_gene_idx, qry_width * sizeof(uint32));
    cudaMalloc((void**)&d_qry_line, qry_width * sizeof(float));
    cudaMalloc((void**)&d_qry_rank, qry_width * sizeof(float));
    cudaMalloc((void**)&d_ref_lines, 1000000 * sizeof(float));
    cudaMalloc((void**)&d_ref_rank, 1000000 * sizeof(float));
    cudaMalloc((void**)&d_score, 1000 * sizeof(float));

    return true;
}

__global__ void get_device_qry_line(uint32* gene_idx, float* qry, const uint32 len, const uint32 gene_len, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        res[tid] = qry[gene_len-gene_idx[tid]-1];//gene_len-1=idx_max    g_idx  int->float res dqry-line descending order
    }
}

__global__ void get_device_ref_lines(uint32* gene_idx, const uint32 gene_len,
    uint32* cell_idx, const uint32 cell_len, float* ref, const uint32 ref_width, 
    const uint32 ref_pitch, float* res)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < cell_len && ny < gene_len)
    {
        float* row_head = (float*)((char*)ref + cell_idx[nx] * ref_pitch);
        res[nx * gene_len + ny] = row_head[ref_width - gene_idx[ny] - 1];
    }
}

__global__ void rankdata(float* qry, const uint32 len, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        int r = 1, s = 0;
        for (int i = 0; i < len; ++i)
        {
            if (qry[tid] == qry[i])
                s += 1;
            else if (qry[tid] > qry[i])
                r += 1;
        }
        res[tid] = r + float(s-1)/2;
    }
}

__global__ void spearman(float* qry, float* ref, const uint32 gene_num, const uint32 cell_num, float* score)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cell_num)
    {
        float mean = (gene_num+1)/2.0;
        float sumxy = 0, sumxx = 0, sumyy = 0;
        for (int i = 0; i < gene_num; ++i)
        {
            float x = qry[i] - mean;
            float y = ref[tid * gene_num + i] - mean;
            sumxy += x * y;
            sumxx += x * x;
            sumyy += y * y;
        }
        float divisor = sqrt(sumxx * sumyy);
        if (divisor != 0)
            score[tid] = sumxy / divisor;
        else
            score[tid] = CUDART_NAN_F;
    }
}

float percentile(vector<float> arr, int len, float p)
{
    if (len <= 1) return arr.front();

    float res;
    std::sort(arr.begin(), arr.begin()+len);

    vector<float> index;
    float step = 1.0/(len-1);
    for (int i = 0; i < len; ++i)
        index.push_back(i*step);
   
    if (p <= index.front())
    {
        res = arr[0];
    }
    else if (index.back() <= p)
    {
        res = arr[len-1];
    }
    else
    {
        auto it = lower_bound(index.begin(), index.end(), p);
        float prevIndex = *(it-1);
        float prevValue = arr.at(it - index.begin()-1);
        float nextValue = arr.at(it - index.begin());
        // linear interpolation
        res = (p - prevIndex) * (nextValue - prevValue) / step + prevValue;
    }
    return res;
}

vector<uint32> finetune_round(float* qry, vector<uint32> top_labels)
{
    // get filtered genes
    // cout<<"top_labels: ";
    // for (auto& label : top_labels)
    //     cout<<label<<" ";
    // cout<<"\ntop label num: "<<top_labels.size() <<" ct_num: "<<ct_num<<endl;
    set<uint32> uniq_genes;
    int gene_thre = round(500 * pow((2/3.0), log2(top_labels.size())));
    // cout<<"gene_thre: "<<gene_thre<<endl;
    
    for (auto& i : top_labels)//??line 159  topl cant be 0??
    {
        for (auto& j : top_labels)
        {
            if (i == j)//same cant be 0?
                continue;
            int pos = h_ctdidx[(i * ct_num + j) * 2];
            int len = h_ctdidx[(i * ct_num + j) * 2 + 1];
            if (len > gene_thre)
                len = gene_thre;
            uniq_genes.insert(h_ctdiff.begin()+pos, h_ctdiff.begin()+pos+len);
            // cout<<"temp uniq genes size: "<<uniq_genes.size()<<endl;
        }
    }
    // cout<<"uniq genes size: "<<uniq_genes.size()<<endl;
    
    vector<uint32> h_gene_idx(uniq_genes.begin(), uniq_genes.end());

    // transfer qry data from cpu to gpu
    cudaMemcpy(d_gene_idx, h_gene_idx.data(), h_gene_idx.size()*sizeof(uint32), cudaMemcpyHostToDevice);
    // cudaMemset(d_qry_line, 0, qry_width * sizeof(float));
    // cudaMemset(d_qry_rank, 0, qry_width * sizeof(float));
    // cudaMemcpy(d_qry_line, h_qry_line.data(), h_qry_line.size()*sizeof(float), cudaMemcpyHostToDevice);
    get_device_qry_line<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_gene_idx, qry, h_gene_idx.size(), qry_width, d_qry_line);

    //check result of get_device_qry_line()
    // vector<float> tmp_qry_line;
    // tmp_qry_line.resize(h_gene_idx.size(), 0);
    // cudaMemcpy(tmp_qry_line.data(), d_qry_line, h_gene_idx.size()*sizeof(float), cudaMemcpyDeviceToHost);
    // cout<<tmp_qry_line.size()<<endl;
    // for (int i = 0; i < tmp_qry_line.size(); ++i)
    //     cout<<tmp_qry_line[i]<<" ";
    // cout<<endl;

    // rank for qry line
    rankdata<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_qry_line, h_gene_idx.size(), d_qry_rank);
    // cudaMemcpy(tmp_qry_line.data(), d_qry_rank, h_gene_idx.size()*sizeof(float), cudaMemcpyDeviceToHost);
    // cout<<tmp_qry_line.size()<<endl;
    // for (int i = 0; i < tmp_qry_line.size(); ++i)
    //     cout<<tmp_qry_line[i]<<" ";
    // cout<<endl;

    // get filtered cells of ref data
    // cudaMemset(d_ref_lines, 0, 1000000 * sizeof(float));
    // cudaMemset(d_ref_rank, 0, 1000000 * sizeof(float));

    vector<float> scores;
    for (auto& label : top_labels)
    {
        uint32 pos = h_ctidx[label * 2];
        uint32 len = h_ctidx[label * 2 + 1];
        
        dim3 blockDim(32, 32);
        dim3 gridDim(len/32+1, h_gene_idx.size()/32+1);
        get_device_ref_lines<<< gridDim, blockDim >>>
            (d_gene_idx, h_gene_idx.size(), d_ctids+pos, len, d_ref, ref_width, pitchref, d_ref_lines);

        // check result of get_device_ref_lines()
        // vector<float> tmp_ref_line;
        // tmp_ref_line.resize(h_gene_idx.size()*len, 0);
        // cudaMemcpy(tmp_ref_line.data(), d_ref_lines, h_gene_idx.size()*len*sizeof(float), cudaMemcpyDeviceToHost);
        // float max_val = 0, total_val = 0;
        // for (int i = 0; i < tmp_ref_line.size(); ++i)
        // {
        //     max_val = max(max_val, tmp_ref_line[i]);
        //     total_val += tmp_ref_line[i];
        //     if (tmp_ref_line[i] > 10)
        //         cout<<i<<","<<tmp_ref_line[i]<<" ";
        // }
        // cout<<max_val<<" "<<total_val<<endl;

        // rank for ref lines
        for (int i = 0; i < len; ++i)
        {
            rankdata<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_ref_lines+i*h_gene_idx.size(), h_gene_idx.size(), d_ref_rank+i*h_gene_idx.size());
        }

        // spearman
        // cudaMemset(d_score, 0, 1000 * sizeof(float));
        spearman<<< len/1024 + 1, 1024 >>>(d_qry_rank, d_ref_rank, h_gene_idx.size(), len, d_score);

        vector<float> h_score;
        h_score.resize(len, 0);
        cudaMemcpy(h_score.data(), d_score, len*sizeof(float), cudaMemcpyDeviceToHost);
        float score = percentile(h_score, len, 0.8);
        // cout<<"score: "<<score<<endl;
        scores.push_back(score);
        
        // cudaMemset(d_ref_lines, 0, h_gene_idx.size() * len * sizeof(float));
    }

    auto ele = std::minmax_element(scores.begin(), scores.end());
    float thre = *ele.second - 0.05;
    vector<uint32> res;
    for (uint32 i = 0; i < scores.size(); ++i)
    {
        if (scores[i] <= *ele.first || scores[i] < thre) continue;
        else res.push_back(top_labels[i]);
    }
    if (res.empty())
        res.push_back(top_labels.front());

    return res;
}

vector<uint32> finetune()
{
    // process each cell
    vector<uint32> res;
    for (int i = 0; i < qry_height; ++i)
    {
        float* qry_head = (float*)((char*)d_qry + i * pitchqry);

        vector<uint32> top_labels;
        uint32 start = i * ct_num;
        for (int pos = 0; pos < ct_num; ++pos)
        {
            if (h_labels.at(start + pos) != 0)
                top_labels.push_back(pos);
        }

        while (top_labels.size() > 1)
        {
            top_labels = finetune_round(qry_head, top_labels);
            // for (auto& label : top_labels)
            //     cout<<label<<endl;
        }
        res.push_back(top_labels.front());
    }
 
    return res;
}
