#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cusingler.cuh"
#include <cub/device/device_radix_sort.cuh>

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
#define LOG
cudaError_t errcode;
cudaStream_t stream;
float* d_ref, *d_qry;
vector<float> h_labels;
uint32 ref_height, ref_width, qry_height, qry_width;
uint32 ref_lines_width;
uint32 ct_num;
uint32* d_ctids;
vector<uint32> h_ctidx;
vector<uint32> h_ctdiff, h_ctdidx;
size_t pitchref;
size_t pitchqry;
size_t pitch_ref_lines;

uint32* d_gene_idx;
float* d_qry_line, *d_qry_rank;

float* d_ref_lines, *d_ref_rank;
float *d_score;
int *d_ref_idx;

cudaError_t err;
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
// unit is MB
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
    ref_lines_width = 0;
    ct_num = 0;
    d_ctids = NULL;
    pitchref=0;
    pitchqry=0;
    pitch_ref_lines=0;
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
    //free(h_qry_idx_sample);


    cudaFree(d_gene_idx);
    cudaFree(d_qry_line);
    cudaFree(d_qry_rank);
    cudaFree(d_ref_lines);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    return true;
}


__global__ void rankdata_pitch_batch_SMEM(float*dataIn,float* dataOut,const int datalen ,const size_t pitch,const int batch)
{
    __shared__ float s_data_line[4096];
    int step=(datalen-1)/blockDim.x+1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<datalen*batch)
    {
        
        float equl_cnt=0;
        float less_cnt=1;
        //copy data to shared memory
        float* dataIn_batch=(float*)((char*)dataIn+blockIdx.x*pitch);
        for(int i=0;i<step;i++)
        {
            if(step*threadIdx.x+i<datalen)
            {
                s_data_line[step*threadIdx.x+i]=dataIn_batch[step*threadIdx.x+i];
                
            }
        }

        __syncthreads();
        //get rank
        for(int i=0;i<step;i++)
        {
            if(step*threadIdx.x+i<datalen)
            {
                for(int j=0;j<datalen;j++)
                {
                    if(s_data_line[j]<s_data_line[step*threadIdx.x+i])
                    {
                        less_cnt++;
                    }
                    else if(s_data_line[j]==s_data_line[step*threadIdx.x+i])
                    {
                        equl_cnt++;
                    }
                }
            dataOut[blockIdx.x*datalen+step*threadIdx.x+i]=less_cnt+(equl_cnt-1)/2.0;
            }
        }

    }

}
__global__ void rankdata_pitch(float* dataIn,float* dataOut,const int datalen,const size_t pitch,const int batch)
{   //no shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<datalen*batch)
    {
        int batchid=tid/datalen;
        float equl_cnt=0;
        float less_cnt=1;
        float* dataIn_batch=(float*)((char*)dataIn+batchid*pitch);
        for(int i=0;i<datalen;i++)
        {

                if(dataIn_batch[tid]>dataIn_batch[i])
                {
                    less_cnt++;
                }
                else if(dataIn_batch[tid]==dataIn_batch[i])
                {
                    equl_cnt++;
                }
            
        }
        dataOut[tid]=less_cnt+(equl_cnt-1)/2;
    }
}
//for qry 1 line rank
__global__ void rankdata(float* dataIn,float* dataOut,const int datalen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<datalen)
    {
        float equl_cnt=0;
        float less_cnt=1;
        for(int i=0;i<datalen;i++)
        {        
                if(dataIn[tid]>dataIn[i])
                {
                    less_cnt++;
                }
                else if(dataIn[tid]==dataIn[i])
                {
                    equl_cnt++;
                }
            
        }
        dataOut[tid]=less_cnt+(equl_cnt-1)/2;
    }

    

}


bool copyin(InputData& rawdata, vector<uint32>& ctids, vector<uint32>& ctidx, vector<uint32>& ctdiff, vector<uint32>& ctdidx)
{
    

    ref_height = rawdata.ref_cell_num;
    ref_width = rawdata.ref_gene_num;
    qry_height = rawdata.test_cell_num;
    qry_width = rawdata.test_gene_num;
    ct_num = rawdata.celltypes.size();


    CHECK(cudaStreamCreate(&stream));
   // cout<<"current stream"<<stream<<endl;
    CHECK(cudaMallocPitch((void**)&d_ref,&pitchref,ref_width*sizeof(float),ref_height));
    CHECK(cudaMallocPitch((void**)&d_qry,&pitchqry,qry_width*sizeof(float),qry_height));
    cudaMalloc((void**)&d_ctids, ctids.size() * sizeof(uint32));
    
    cudaMemcpy2DAsync(d_ref,pitchref, rawdata.ref.data(), ref_width * sizeof(float),ref_width * sizeof(float),ref_height,cudaMemcpyHostToDevice,stream);
    cudaMemcpy2DAsync(d_qry,pitchqry, rawdata.test.data(),qry_width * sizeof(float),qry_width * sizeof(float),qry_height,cudaMemcpyHostToDevice,stream);

    h_labels = rawdata.labels;
    //CHECK( cudaMemcpyAsync(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpy(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice));
   

    h_ctidx = ctidx;

    h_ctdiff = ctdiff;
    h_ctdidx = ctdidx;
    cudaStreamSynchronize(stream);
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout<<"used gpu mem(MB): "<<getUsedMem()<<std::endl;

    cudaMalloc((void**)&d_gene_idx, qry_width * sizeof(uint32));
    cudaMalloc((void**)&d_qry_line, qry_width * sizeof(float));
    cudaMalloc((void**)&d_qry_rank, qry_width * sizeof(float));
    
    //h_genidx.size() <4096
    ref_lines_width=4096;
    CHECK(cudaMallocPitch((void**)&d_ref_lines,&pitch_ref_lines,ref_lines_width*sizeof(float),ref_height));

    
    cudaMalloc((void**)&d_ref_rank, 100000000 * sizeof(float));
    cudaMalloc((void**)&d_score, 100000 * sizeof(float));
    cudaMalloc((void**)&d_ref_idx, 100000000 * sizeof(int));

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

__global__ void get_device_ref_pitch(uint32* gene_idx, const uint32 gene_len,
    uint32* cell_idx, const uint32 cell_len, float* ref, const uint32 ref_width, 
    const uint32 ref_pitch,const uint32 ref_lines_width,const uint32 ref_lines_pitch , float* ref_lines)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < cell_len && ny < gene_len)
    {
        float* row_head = (float*)((char*)ref + (uint64)(cell_idx[nx]) * ref_pitch);
       // float* row_head_lines = (float*)((char*)ref_lines + (uint64)(cell_idx[nx]) * ref_lines_pitch);
       float* row_head_lines = (float*)((char*)ref_lines + (uint64)(nx) * ref_lines_pitch);
       row_head_lines[ny] = row_head[ref_width - gene_idx[ny] - 1];
        
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
        float* row_head = (float*)((char*)ref + (uint64)(cell_idx[nx]) * ref_pitch);
        res[nx * gene_len + ny] = row_head[ref_width - gene_idx[ny] - 1];
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
    clock_t t0,t1;
    float time;
    // get filtered genes
    // cout<<"top_labels: ";
    // for (auto& label : top_labels)
    //     cout<<label<<" ";
    // cout<<"\ntop label num: "<<top_labels.size() <<" ct_num: "<<ct_num<<endl;
    set<uint32> uniq_genes;
    int gene_thre = round(500 * pow((2/3.0), log2(top_labels.size())));
    // cout<<"gene_thre: "<<gene_thre<<endl;
   // t0=clock();
    
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
   // t1=clock();
   // time=(float)(t1-t0)/CLOCKS_PER_SEC;
   // cout<<"get uniq_genes time: "<<time<<endl;
  
    // t0=clock();
    vector<uint32> h_gene_idx(uniq_genes.begin(), uniq_genes.end());
    
    // transfer qry data from cpu to gpu
    CHECK(cudaMemcpy(d_gene_idx, h_gene_idx.data(), h_gene_idx.size()*sizeof(uint32), cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(d_qry_idx,d_qry_idx_sample,10000*sizeof(int),cudaMemcpyDeviceToDevice));//copy oringin idx to current idx for sort
    get_device_qry_line<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_gene_idx, qry, h_gene_idx.size(), qry_width, d_qry_line);

    rankdata<<<h_gene_idx.size()/1024 + 1, 1024>>>(d_qry_line, d_qry_rank, h_gene_idx.size());
    // thrust::device_ptr<float> dev_ptr(d_qry_rank);
    // for (size_t i = 0; i < 10; i++)
    // {
    //     cout<<dev_ptr[i]<<" ";
    // }


    cudaMemset2D(d_ref_lines,pitch_ref_lines,0, ref_lines_width *sizeof(float),ref_height);
     //cudaMemset(d_ref_lines, 0, 1000000 * sizeof(float));
    cudaMemset(d_ref_rank, 0, 1000000 * sizeof(float));
    
    vector<float> scores;    
    for (auto& label : top_labels)
    {
      
 
        uint32 pos = h_ctidx[label * 2];
        uint32 len = h_ctidx[label * 2 + 1];
       // cout<<" len: "<<len<<endl;
        dim3 blockDim(32, 32);
        dim3 gridDim((len-1)/32+1, (h_gene_idx.size()-1)/32+1);
   

        get_device_ref_pitch<<<gridDim, blockDim>>>(d_gene_idx, h_gene_idx.size(), d_ctids+pos, len, d_ref, ref_width, pitchref,ref_lines_width,pitch_ref_lines,d_ref_lines);
        err = cudaGetLastError();
        if (err != cudaSuccess )
        {
            printf("get_device_ref_pitch CUDA Error: %s\n", cudaGetErrorString(err));
        }   

    //*****************rank data

    rankdata_pitch<<<len,512>>>(d_ref_lines,d_ref_rank,h_gene_idx.size(),pitch_ref_lines,len);
        //rankdata_pitch_batch_SMEM<<<len,512>>>(d_ref_lines,d_ref_rank,h_gene_idx.size(),pitch_ref_lines,len);
  
        err = cudaGetLastError();
        if (err != cudaSuccess )
        {
            printf("rankdata_batch CUDA Error: %s\n", cudaGetErrorString(err));
        }        
    //*****************rank data 
        // spearman
        cudaMemset(d_score, 0, 1000 * sizeof(float));

        spearman<<< len/1024 + 1, 1024 >>>(d_qry_rank, d_ref_rank, h_gene_idx.size(), len, d_score);
 
        vector<float> h_score;
        h_score.resize(len, 0);
        CHECK( cudaMemcpy(h_score.data(), d_score, len*sizeof(float), cudaMemcpyDeviceToHost));


        float score = percentile(h_score, len, 0.8);
        // cout<<"score: "<<score<<endl;
        scores.push_back(score);
        
      
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
    // t1=clock();
    // time=(float)(t1-t0)/CLOCKS_PER_SEC;
    // cout<<"rest etc : "<<time<<endl;
    return res;
}

vector<uint32> finetune()
{
    // process each cell
    vector<uint32> res;
    cout<<"cell num:"<<qry_height<<endl;
    // clock_t startT,endT;
    // float timecell;
    // for (int i = 0; i < 1; ++i)
    //for (int i = 26; i < 27; ++i)
    for (int i = 0; i < qry_height; ++i)
    {
        // startT=clock();
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
           // cout<<"top_labels size"<<top_labels.size()<<endl;
            top_labels = finetune_round(qry_head, top_labels);
            // for (auto& label : top_labels)
            //     cout<<label<<endl;
        }
        res.push_back(top_labels.front());
        if (i % 100 == 0)
        {
            auto now = std::chrono::system_clock::now();
            std::time_t curr_time = std::chrono::system_clock::to_time_t(now);
            cout<<std::ctime(&curr_time)<<"\tprocessed "<<i<<" cells"<<endl;
        }

        // endT=clock();
        // timecell=(float)(endT-startT)/CLOCKS_PER_SEC;
        // cout<<"cell"<<i<<"procTime:"<<timecell<<endl;

        // if(i==199)
        //     break;
        
    }
 
    return res;
}
