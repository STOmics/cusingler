#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cusingler.cuh"
#include "timer.h"
#include <cub/device/device_radix_sort.cuh>

#include <stdlib.h>
#include <stdio.h>

#include <cmath>
#include <chrono>
#include <thread>
#include <iostream>
#include <set>
#include <algorithm>
#include <numeric>

#include "cuda_runtime.h"
#include "math_constants.h"
#define LOG
cudaError_t errcode;
cudaStream_t stream;
uint16* d_ref, *d_qry;
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

uint32* d_gene_idx, *d_cell_idx;
uint16 *d_ref_lines, *d_qry_line;
float *d_ref_rank, *d_qry_rank;
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
    cudaFree(d_cell_idx);
    cudaFree(d_qry_line);
    cudaFree(d_qry_rank);
    cudaFree(d_ref_lines);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    return true;
}


// __global__ void rankdata_pitch_batch_SMEM(float*dataIn,float* dataOut,const int datalen ,const size_t pitch,const int batch)
// {
//     __shared__ float s_data_line[4096];
//     int step=(datalen-1)/blockDim.x+1;
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid<datalen*batch)
//     {
        
//         float equl_cnt=0;
//         float less_cnt=1;
//         //copy data to shared memory
//         float* dataIn_batch=(float*)((char*)dataIn+blockIdx.x*pitch);
//         for(int i=0;i<step;i++)
//         {
//             if(step*threadIdx.x+i<datalen)
//             {
//                 s_data_line[step*threadIdx.x+i]=dataIn_batch[step*threadIdx.x+i];
                
//             }
//         }

//         __syncthreads();
//         //get rank
//         for(int i=0;i<step;i++)
//         {
//             if(step*threadIdx.x+i<datalen)
//             {
//                 for(int j=0;j<datalen;j++)
//                 {
//                     if(s_data_line[j]<s_data_line[step*threadIdx.x+i])
//                     {
//                         less_cnt++;
//                     }
//                     else if(s_data_line[j]==s_data_line[step*threadIdx.x+i])
//                     {
//                         equl_cnt++;
//                     }
//                 }
//             dataOut[blockIdx.x*datalen+step*threadIdx.x+i]=less_cnt+(equl_cnt-1)/2.0;
//             }
//         }

//     }

// }

__global__ void rankdata_pitch(uint16* dataIn,float* dataOut,const int datalen,const size_t pitch,const int batch)
{   //no shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<datalen*batch)
    {
        int batchid=tid/datalen;
        float equl_cnt=0;
        float less_cnt=1;
        uint16* dataIn_batch=(uint16*)((char*)dataIn+batchid*pitch);
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



// __global__ void rankdata_pitch(float* dataIn,float* dataOut,const int datalen,const size_t pitch,const int batch)
// {   //no shared memory
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid<datalen*batch)
//     {
//         int batchid=tid/datalen;
//         float equl_cnt=0;
//         float less_cnt=1;
//         float* dataIn_batch=(float*)((char*)dataIn+batchid*pitch);
//         for(int i=0;i<datalen;i++)
//         {

//                 if(dataIn_batch[tid]>dataIn_batch[i])
//                 {
//                     less_cnt++;
//                 }
//                 else if(dataIn_batch[tid]==dataIn_batch[i])
//                 {
//                     equl_cnt++;
//                 }
            
//         }
//         dataOut[tid]=less_cnt+(equl_cnt-1)/2;
//     }
// }
//for qry 1 line rank
__global__ void rankdata(uint16* dataIn,float* dataOut,const int datalen)
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
//
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




bool copyin(InputData& rawdata, vector<uint32>& ctids, vector<uint32>& ctidx, vector<uint32>& ctdiff, vector<uint32>& ctdidx,
    vector<uint16>& ref, vector<uint16>& qry)
{
    ref_height = rawdata.ref_cell_num;
    ref_width = rawdata.ref_gene_num;
    qry_height = rawdata.test_cell_num;
    qry_width = rawdata.test_gene_num;
    ct_num = rawdata.celltypes.size();


    CHECK(cudaStreamCreate(&stream));
   // cout<<"current stream"<<stream<<endl;
    CHECK(cudaMallocPitch((void**)&d_ref,&pitchref,ref_width*sizeof(uint16),ref_height));
    CHECK(cudaMallocPitch((void**)&d_qry,&pitchqry,qry_width*sizeof(uint16),qry_height));
    cudaMalloc((void**)&d_ctids, ctids.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctidx, ctidx.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctdiff, ctdiff.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctdidx, ctdidx.size() * sizeof(uint32));

    cudaMemcpy2DAsync(d_ref,pitchref, ref.data(), ref_width * sizeof(uint16),ref_width * sizeof(uint16),ref_height,cudaMemcpyHostToDevice,stream);
    cudaMemcpy2DAsync(d_qry,pitchqry, qry.data(),qry_width * sizeof(uint16),qry_width * sizeof(uint16),qry_height,cudaMemcpyHostToDevice,stream);
   

    // cudaMemcpyAsync(d_ref, rawdata.ref.data(), ref_height * ref_width * sizeof(float), cudaMemcpyHostToDevice,stream);
    // cudaMemcpyAsync(d_qry, rawdata.test.data(), qry_height * qry_width * sizeof(float), cudaMemcpyHostToDevice,stream);
    // // cudaMemcpy(d_labels, rawdata.labels.data(), qry_height * ct_num * sizeof(float), cudaMemcpyHostToDevice);
    h_labels = rawdata.labels;
    //CHECK( cudaMemcpyAsync(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpy(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice));
   

    h_ctidx = ctidx;

    h_ctdiff = ctdiff;
    h_ctdidx = ctdidx;
    cudaStreamSynchronize(stream);
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    

    cudaMalloc((void**)&d_gene_idx, qry_width * sizeof(uint32));
    cudaMalloc((void**)&d_cell_idx, ref_height * sizeof(uint32));
    cudaMalloc((void**)&d_qry_line, qry_width * sizeof(uint16));
    cudaMalloc((void**)&d_qry_rank, qry_width * sizeof(float));
    
    //h_genidx.size() <4096
    ref_lines_width=4096;
    CHECK(cudaMallocPitch((void**)&d_ref_lines,&pitch_ref_lines,ref_lines_width*sizeof(uint16),ref_height));

    
    cudaMalloc((void**)&d_ref_rank, 100000000 * sizeof(float));
    cudaMalloc((void**)&d_score, 100000000 * sizeof(float));

    std::cout<<"used gpu mem(MB): "<<getUsedMem()<<std::endl;

    return true;
}



__global__ void get_device_qry_line(uint32* gene_idx, uint16* qry, const uint32 len, const uint32 gene_len, uint16* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        res[tid] = qry[gene_len-gene_idx[tid]-1];//gene_len-1=idx_max    g_idx  int->float res dqry-line descending order
    }
}

__global__ void get_device_ref_pitch(uint32* gene_idx, const uint32 gene_len,
    uint32* cell_idx, const uint32 cell_len, uint16* ref, const uint32 ref_width, 
    const size_t ref_pitch,const uint32 ref_lines_width,const size_t ref_lines_pitch , uint16* ref_lines)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < cell_len && ny < gene_len)
    {
        uint16* row_head = (uint16*)((char*)ref + (uint64)(cell_idx[nx]) * ref_pitch);
       // float* row_head_lines = (float*)((char*)ref_lines + (uint64)(cell_idx[nx]) * ref_lines_pitch);
       uint16* row_head_lines = (uint16*)((char*)ref_lines + (uint64)(nx) * ref_lines_pitch);
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


__global__ void rankdata(uint16* qry, const uint32 len, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        float r = 1, s = 0;
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

// basic rankdata method using bincount
__global__ void rankdata_bin(uint16* qry, const uint32 cols, const uint32 rows, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows)
    {
        uint16* q = qry + tid*cols;
        float*  r = res + tid*cols;
        // travel for getting bin count
        uint16 bins[80] = {0};
        for (int i = 0; i < cols; ++i)
            bins[q[i]]++;

        // calculate real rank
        float ranks[80];
        float start = 0;
        for (int i = 0; i < 80; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[i] = start + (bins[i]+1)*0.5;
            start += bins[i];
        }

        // assign rank value
        for (int i = 0; i < cols; ++i)
            r[i] = ranks[q[i]];
    }
}

// using shared memory
__global__ void rankdata_bin2(uint16* qry, const uint32 cols, const uint32 rows, float* res)
{
    __shared__ uint16 bins[80*64];
    __shared__ float ranks[80*64];
    int bid = threadIdx.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows)
    {
        for (int i = 0; i < 80; ++i)
        {
            bins[bid*80+i] = 0;
        }

        uint16* q = qry + tid*cols;
        float*  r = res + tid*cols;
        // travel for getting bin count
        for (int i = 0; i < cols; ++i)
            bins[bid*80+q[i]]++;

        // calculate real rank
        float start = 0;
        for (int i = 0; i < 80; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[bid*80+i] = start + (bins[bid*80+i]+1)*0.5;
            start += bins[bid*80+i];
        }

        // assign rank value
        for (int i = 0; i < cols; ++i)
            r[i] = ranks[bid*80+q[i]];
    }
}

// one block threads for each cell
__global__ void rankdata_bin3(uint16* qry, const uint32 cols, const uint32 rows, float* res)
{
    __shared__ int bins[128];
    __shared__ float ranks[128];
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int step = cols / 64;
    if (cols % 64 != 0)
        step++;

    if (bid < rows)
    {
        bins[tid*2] = 0;
        bins[tid*2+1] = 0;
        __syncthreads();

        uint16* q = qry + bid*cols;
        float*  r = res + bid*cols;
        // travel for getting bin count
        for (int i = tid*step; i < (tid+1)*step; ++i)
        {
            if (i < cols)
                atomicAdd(&bins[q[i]], 1);
        }
        __syncthreads();

        // calculate real rank
        if (tid == 0)
        {
            int start = 0;
            for (int i = 0; i < 128; ++i)
            {
                // if (bins[i] == 0) continue;
                ranks[i] = start + (bins[i]+1)*0.5;
                start += bins[i];
            }
        }
        __syncthreads();

        // assign rank value
        for (int i = tid*step; i < (tid+1)*step; ++i)
        {
            if (i < cols)
                r[i] = ranks[q[i]];
        }
    }
}

__global__ void spearman_reduce(float* qry, float* ref, const uint32 gene_num, const uint32 cell_num, float* score)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float mean = (gene_num+1)/2.0;
    int step = gene_num / 128;
    if (gene_num % 128 != 0)
        step++;

    __shared__ float sumxy[128];
    __shared__ float sumxx[128];
    __shared__ float sumyy[128];

    sumxy[tid] = 0;
    sumxx[tid] = 0;
    sumyy[tid] = 0;
    
    for (int i = tid*step; i < (tid+1)*step; ++i)
    {
        if (i < gene_num)
        {
            float x = qry[i] - mean;
            float y = ref[bid * gene_num + i] - mean;
            sumxy[tid] += x * y;
            sumxx[tid] += x * x;
            sumyy[tid] += y * y;
        }
    }
    __syncthreads();
    
    for (int stride = 1; stride < 128; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            sumxy[tid] += sumxy[tid + stride];
            sumxx[tid] += sumxx[tid + stride];
            sumyy[tid] += sumyy[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float divisor = sqrt(sumxx[0] * sumyy[0]);
        if (divisor != 0)
            score[bid] = sumxy[0] / divisor;
        else
            score[bid] = CUDART_NAN_F;
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

vector<uint32> finetune_round(uint16* qry, vector<uint32> top_labels)
{


    set<uint32> uniq_genes;
    int gene_thre = round(500 * pow((2/3.0), log2(top_labels.size())));

    
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

    vector<uint32> h_gene_idx(uniq_genes.begin(), uniq_genes.end());

    // transfer qry data from cpu to gpu
    CHECK(cudaMemcpy(d_gene_idx, h_gene_idx.data(), h_gene_idx.size()*sizeof(uint32), cudaMemcpyHostToDevice));
    get_device_qry_line<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_gene_idx, qry, h_gene_idx.size(), qry_width, d_qry_line);
           err = cudaGetLastError();
        if (err != cudaSuccess )
        {
            printf("get_device_qry_line CUDA Error: %s\n", cudaGetErrorString(err));
        }     
    // get rank of qry data
    // if mod0
    rankdata<<<h_gene_idx.size()/1024 + 1, 1024>>>(d_qry_line, d_qry_rank, h_gene_idx.size());
               err = cudaGetLastError();
        if (err != cudaSuccess )
        {
            printf("rank CUDA Error: %s\n", cudaGetErrorString(err));
        }     
    // if mod1
    //bin
  
    vector<pair<size_t, size_t>> temp;
    size_t total_len = 0;
    for (auto& label : top_labels)
    {
        uint32 pos = h_ctidx[label * 2];
        uint32 len = h_ctidx[label * 2 + 1];
        // cout<<label<<" "<<pos<<" "<<len<<endl;
        if (temp.empty() || (temp.back().first + temp.back().second) != pos)
        {
            temp.push_back({pos,len});
            total_len += len;
        }
        else
        {
            temp.back().second += len;
            total_len += len;
        }
    }
    vector<uint32> h_cell_idx(total_len);
    total_len = 0;
    //for (auto& [pos, len] : temp)
    for(auto&tmp:temp)
    {
        size_t pos=tmp.first;
        size_t len=tmp.second;
        std::iota(h_cell_idx.begin()+total_len, h_cell_idx.begin()+total_len+len, pos);
        // cout<<total_len<<" "<<total_len+len<<" "<<pos<<endl;
        total_len += len;
    }
    CHECK(cudaMemcpy(d_cell_idx, h_cell_idx.data(), h_cell_idx.size()*sizeof(uint32), cudaMemcpyHostToDevice));
    dim3 blockDim(1, 512);
    dim3 gridDim(h_cell_idx.size(), (h_gene_idx.size()-1)/512+1);
    get_device_ref_pitch<<<gridDim,blockDim>>>(d_gene_idx,h_gene_idx.size(),d_cell_idx,h_cell_idx.size(),
                                                d_ref,ref_width,pitchref,
                                                ref_lines_width,pitch_ref_lines,d_ref_lines);
    CHECK(cudaGetLastError());

    //if mod 0
    cout<<"h_gene_idx.size()"<<h_gene_idx.size()<<endl;
    cout<<"total_len"<<total_len;
    getchar()   ;
    // uint16* row_head = (uint16*)((char*)d_ref_lines +  pitch_ref_lines);
      
    // thrust::device_ptr<uint16> dev_ptr0(row_head);
    //     for(int i=0;i<h_gene_idx.size();i++)
    // {   
    //     cout<<dev_ptr0[i]<<" ";               
    // }
    // getchar()   ; 
    dim3 gridDim0(h_cell_idx.size(), (h_gene_idx.size()-1)/512+1);
    rankdata_pitch<<<h_cell_idx.size(),512>>>(d_ref_lines,d_ref_rank,h_gene_idx.size(),pitch_ref_lines,h_cell_idx.size());
    CHECK(cudaGetLastError());
    
    // thrust::device_ptr<float> dev_ptr(d_score);
    //    for(int i=0;i<h_gene_idx.size();i++)
    // {
    // cout<<dev_ptr[i]<<" ";
    // }
    // getchar()   ;
    thrust::device_ptr<float> dev_ptr2(d_ref_rank);
    thrust::device_ptr<float> dev_ptr3(d_qry_rank);
 
    for(int i=0;i<h_gene_idx.size()*3;i++)
    {   
        cout<<dev_ptr2[i]<<" ";               
    }
    getchar()   ;
    for(int i=0;i<h_gene_idx.size();i++)
    {   
        cout<<dev_ptr3[i]<<" ";               
    }
    getchar()   ;
    //if mod 1
    //bin

     //spearman
    spearman_reduce<<< total_len, 128 >>>(d_qry_rank, d_ref_rank, h_gene_idx.size(), total_len, d_score);

 cudaError_t err = cudaGetLastError();
 if( err != cudaSuccess) std::cout << "Error: " << cudaGetErrorString(err) << std::endl;  CHECK(cudaGetLastError());

    vector<float> h_score;
    h_score.resize(total_len, 0);
    cudaMemcpy(h_score.data(), d_score, total_len*sizeof(float), cudaMemcpyDeviceToHost);
    uint32 start = 0;
    total_len = 0;
    vector<float> scores;
    for (auto& label : top_labels)
    {
        uint32 len = h_ctidx[label * 2 + 1];
        total_len += len;
        
        vector<float> tmp(h_score.begin()+start, h_score.begin()+total_len);
        float score = percentile(tmp, len, 0.8);
        // cout<<label<<" score: "<<score<<endl;
        scores.push_back(score);
        start += len;
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
    Timer timer("ms");
    // process each cell
    vector<uint32> res;
    cout<<"cell num:"<<qry_height<<endl;
    // clock_t startT,endT;
    // float timecell;
    // for (int i = 0; i < 1; ++i)
    //for (int i = 26; i < 27; ++i)
    for (int i = 0; i < qry_height; ++i)
    {
        uint16* qry_head = (uint16*)((char*)d_qry + i * pitchqry);

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
            cout<<"processed "<<i<<" cells cost time(ms): "<<timer.toc()<<endl;
        }

        // endT=clock();
        // timecell=(float)(endT-startT)/CLOCKS_PER_SEC;
        // cout<<"cell"<<i<<"procTime:"<<timecell<<endl;

        // if(i==199)
        //     break;
        
    }
 
    return res;
}
