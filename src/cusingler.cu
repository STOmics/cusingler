
#include "cusingler.cuh"

#include <stdlib.h>
#include <stdio.h>

#include <cmath>
#include <chrono>
#include <thread>
#include <iostream>
#include <set>

#include "cuda_runtime.h"
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
// unit is MB
#define CHECK(call)                                                       \
{                                                                         \
   const cudaError_t error = call;                                        \
   if (error != cudaSuccess)                                              \
   {                                                                      \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
      exit(1);                                                            \
   }                                                                      \
}

uint32 getUsedMem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (total-free)/1024/1024;
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

    
    CHECK(cudaMallocPitch((void**)&d_ref,&pitchref,ref_width*sizeof(float),ref_height));
    CHECK(cudaMallocPitch((void**)&d_qry,&pitchqry,qry_width*sizeof(float),qry_height));
    
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
    //CHECK( cudaMemcpyAsync(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice,stream));
    CHECK( cudaMemcpy(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice));
   
    // cudaMemcpy(d_ctidx, ctidx.data(), ctidx.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    h_ctidx = ctidx;
    // cudaMemcpy(d_ctdiff, ctdiff.data(), ctdiff.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_ctdidx, ctdidx.data(), ctdidx.size() * sizeof(uint32), cudaMemcpyHostToDevice);
    h_ctdiff = ctdiff;
    h_ctdidx = ctdidx;
    cudaStreamSynchronize(stream);
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout<<"used gpu mem(MB): "<<getUsedMem()<<std::endl;

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
    uint32* cell_idx, const uint32 cell_len, float* ref, const uint32 pitch,const uint32 ref_width, 
    float* res)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < cell_len && ny < gene_len)
    {
        res[nx * gene_len + ny] = ref[cell_idx[nx] * (pitch/sizeof(float)) + ref_width - gene_idx[ny] - 1 ];
    }
}

__global__ void rankdata(float* qry,float* d_rankout, const uint32 len)
{
    //int nx = blockIdx.x * blockDim.x + threadIdx.x;
    //int ny = blockIdx.y * blockDim.y + threadIdx.y;
    int tidx=threadIdx.x;
    
    int r = 1, s = 0;

    if(tidx<len)
    {
        for (int i=0;i<len;i++)
        {
            if(i!=tidx)     //i!=j
            {
            if(qry[i]<qry[tidx])
                r+=qry[i];
            if(qry[i]==qry[tidx])
                s+=qry[i] ;   
            }
        }
        d_rankout[tidx]=r+s*0.5;
    }

}

__global__ void spearman(float* qry, float* ref, const uint32 gene_num, const uint32 cell_num, float* score)
{
    // TODO
}

bool finetune_round(float* qry, float* labels, int line_num)
{
    // get filtered genes
    vector<uint32> top_labels;
    uint32 start = line_num * ct_num;
    for (int i = 0; i < ct_num; ++i)
    {
        if (h_labels.at(start + i) != 0)
            top_labels.push_back(i);
    }
    cout<<"top_labels: ";
    for (auto& label : top_labels)
        cout<<label<<" ";
    cout<<"\ntop label num: "<<top_labels.size() <<" ct_num: "<<ct_num<<endl;
    set<uint32> uniq_genes;
    int gene_thre = round(500 * pow((2/3.0), log2(top_labels.size())));
    cout<<"gene_thre: "<<gene_thre<<endl;
    
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
    cout<<"uniq genes size: "<<uniq_genes.size()<<endl;
    
    vector<uint32> h_gene_idx(uniq_genes.begin(), uniq_genes.end());

    // transfer qry data from cpu to gpu
    uint32* d_gene_idx;
    cudaMalloc((void**)&d_gene_idx, h_gene_idx.size() * sizeof(uint32));
    cudaMemcpy(d_gene_idx, h_gene_idx.data(), h_gene_idx.size()*sizeof(uint32), cudaMemcpyHostToDevice);

    float* d_qry_line;
    cudaMalloc((void**)&d_qry_line, qry_width * sizeof(float));
    cudaMemset(d_qry_line, 0, qry_width * sizeof(float));
    // cudaMemcpy(d_qry_line, h_qry_line.data(), h_qry_line.size()*sizeof(float), cudaMemcpyHostToDevice);
    get_device_qry_line<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_gene_idx, qry, h_gene_idx.size(), qry_width, d_qry_line);

    //check result of get_device_qry_line()
    vector<float> tmp_qry_line;
    tmp_qry_line.resize(h_gene_idx.size(), 0);
    CHECK(cudaMemcpy(tmp_qry_line.data(), d_qry_line, h_gene_idx.size()*sizeof(float), cudaMemcpyDeviceToHost));
    cout<<tmp_qry_line.size()<<endl;
    for (int i = 0; i < tmp_qry_line.size(); ++i)
        cout<<tmp_qry_line[i]<<" ";
    cout<<endl;
    float * d_rank;
    CHECK(cudaMalloc((void**)&d_rank,h_gene_idx.size()*sizeof(float)));
    // rank for qry line
    dim3 blockDim(1024);
    dim3 gridDim((h_gene_idx.size()-1)/1024+1);
    rankdata<<< gridDim, blockDim >>>(d_qry_line,d_rank, h_gene_idx.size());
    //check result of rankdata()
    CHECK(cudaMemcpy(tmp_qry_line.data(), d_rank, h_gene_idx.size()*sizeof(float), cudaMemcpyDeviceToHost));
    cout<<"rankresult:"<<endl;
    cout<<tmp_qry_line.size()<<endl;
    for (int i = 0; i < tmp_qry_line.size(); ++i)
        cout<<tmp_qry_line[i]<<" ";
    cout<<endl;


    // get filtered cells of ref data
    float* d_ref_lines;
    CHECK(cudaMalloc((void**)&d_ref_lines, 1000000 * sizeof(float)));
    CHECK(cudaMemset(d_ref_lines, 0, 1000000 * sizeof(float)));
  

    for (auto& label : top_labels)
    {
      
 
        uint32 pos = h_ctidx[label * 2];
        uint32 len = h_ctidx[label * 2 + 1];
        
        dim3 blockDim1(32, 32);
        dim3 gridDim1((len-1)/32+1, (h_gene_idx.size()-1)/32+1);
        get_device_ref_lines<<< gridDim1, blockDim1 >>>
            (d_gene_idx, h_gene_idx.size(), d_ctids+pos, len, d_ref, pitchref,ref_width, d_ref_lines);
        /*
          ********d_ref is pitched mem***********
          data[i][j]=d_ref+i*pitch/sizeof(datatype)+j
        */
        //check result of get_device_ref_lines()
        vector<float> tmp_ref_line;
        tmp_ref_line.resize(h_gene_idx.size()*len, 0);
        CHECK( cudaMemcpy(tmp_ref_line.data(), d_ref_lines, h_gene_idx.size()*len*sizeof(float), cudaMemcpyDeviceToHost));
      
        cout<<"h_gene_idx len: "<<h_gene_idx.size()<<endl;
        CHECK( cudaMemcpy(tmp_ref_line.data(), d_ref_lines, h_gene_idx.size()*len*sizeof(float), cudaMemcpyDeviceToHost));
        cout<<tmp_ref_line.size()<<endl;
        float max_val = 0, total_val = 0;
        for (int i = 0; i < tmp_ref_line.size(); ++i)
        {
            max_val = max(max_val, tmp_ref_line[i]);
            total_val += tmp_ref_line[i];
        }
        cout<<max_val<<" "<<total_val<<endl;

        // rank for ref lines
        for (int i = 0; i < len; ++i)
        {
           // rankdata<<< 1, 1 >>>(d_ref_lines+i*h_gene_idx.size(), h_gene_idx.size());
            break;
        }

        // spearman
        float score;
        spearman<<<1, 1>>>(d_qry_line, d_ref_lines, h_gene_idx.size(), len, &score);
        
        cudaMemset(d_ref_lines, 0, h_gene_idx.size() * len * sizeof(float));

        // test 
        break;
    }

    // clear resources
    cudaFree(d_gene_idx);
    cudaFree(d_qry_line);
    cudaFree(d_ref_lines);

    return true;
}

bool finetune()
{
    // process each cell
    for (int i = 0; i < qry_height; ++i)
    {
        //finetune_round(d_qry+i*qry_width, NULL, i);
        cout<<"p_qry:"<< pitchqry<<"qry_width:"<<qry_width<<endl;
        finetune_round(d_qry+i*pitchqry/sizeof(float), NULL, i);
        break;
    }
 
    return true;
}
