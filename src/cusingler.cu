#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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
#define LOG
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
int* h_qry_idx_sample;//  to be realesed
int* d_qry_idx_sample;
int* d_qry_idx;    //idx for pair sort
float* d_ref_lines, *d_ref_rank;
float *d_score;

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
    free(h_qry_idx_sample);


    cudaFree(d_gene_idx);
    cudaFree(d_qry_line);
    cudaFree(d_qry_rank);
    cudaFree(d_ref_lines);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    return true;
}


bool rank_by_unique(float* dataInOut ,int* dataIdx,int datalen)
{
    //thrust::device_ptr<float> thrust_InOut(dataInOut);
    // for(int i=0;i<datalen;i++)
    // {
    //     cout<<"data: "<<thrust_InOut[i]<<endl;
    // }
    thrust::device_vector<float> datavec(dataInOut,dataInOut+datalen);
    thrust::device_vector<float> out(datalen) ;
    thrust::sort(datavec.begin(),datavec.end());
    // for(int i=0;i<datalen;i++)
    // {
    //     cout<<"data: "<<datavec[i]<<endl;
    // }
    
    auto eend= thrust::unique_copy(datavec.begin(), datavec.end(), out.begin());
    auto num_unique = thrust::distance(out.begin(), eend);

     cout<<"uniq cnt"<<num_unique<<endl;
     for(int i=0;i<num_unique;i++)
     {
         cout<<"uniq data: "<<out[i]<<endl;
     }
    return true;
}


__global__ void sortrank(float* dataInOut, int* dataIdx,int datalen)
{   
    uint32 tid=blockIdx.x*blockDim.x+threadIdx.x;
    uint32 tid_idx;
    uint32 offset=0;
    uint32 num_swaps;
    uint32 tid_idx_max=datalen-1;
do 
{
    num_swaps=0;
    tid_idx=tid*2+offset;
    if(tid_idx<tid_idx_max)
    {
       
        if(dataInOut[tid_idx]>dataInOut[tid_idx+1])
        {
            float tmp=dataInOut[tid_idx];
            dataInOut[tid_idx]=dataInOut[tid_idx+1];
            dataInOut[tid_idx+1]=tmp;
            int tmpidx=dataIdx[tid_idx];
            dataIdx[tid_idx]=dataIdx[tid_idx+1];
            dataIdx[tid_idx+1]=tmpidx;
            num_swaps=1;
        }
        
    }
    offset=1-offset;
}while(__syncthreads_count(num_swaps)!=0);

}
//get rank after sort
__global__ void sumrank(float* dataInOut,int* rankidx,const int datalen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
   // __shared__ float s_rank[1024] ;
    if(tid<datalen)
    {
        int idxL=tid;
        int idxR=tid;
        while(idxL>0&(dataInOut[idxL]==dataInOut[idxL-1]))
            idxL-=1;
        do
        {
            idxR+=1;
        }while (idxR!=datalen&(dataInOut[idxR]==dataInOut[idxR-1]));
     //   s_rank[threadIdx.x]=1+idxL+float(idxR-idxL-1)/2;
      float tmp=1+idxL+float(idxR-idxL-1)/2;
        __syncthreads();
    //    dataInOut[rankidx[tid]]=s_rank[threadIdx.x];
    dataInOut[rankidx[tid]]=tmp;
    }

}
bool sort_by_idx1(float* dataInOut ,int* dataIdx,int datalen)
{
    //get thrust ptr
    // std::cout<<"used gpu mem(MB): "<<getUsedMem()<<std::endl;
    thrust::device_ptr<float> thrust_InOut(dataInOut);
    thrust::device_ptr<int> thrust_Idx(dataIdx);     
    // getchar();
    // std::cout<<"used gpu mem(MB): "<<getUsedMem()<<std::endl;  
    // getchar();
    
    //
    // thrust::sort_by_key(thrust_InOut,thrust_InOut+datalen,thrust_Idx);
    // for(int i=0;i<datalen;i++)
    // {
    //     cout<<"data: "<<thrust_InOut[i]<<"idx："<<thrust_Idx[i]<<endl;
    // }
    //get rank
    sumrank<<<datalen/256+1,256>>>(dataInOut,dataIdx,datalen);
    // for(int i=0;i<datalen;i++)
    // {
    //     cout<<"data: "<<thrust_InOut[i]<<"idx："<<thrust_Idx[i]<<endl;
    // }
    // getchar();
    return true;
}
bool sort_by_idx(float* dataInOut ,int* dataIdx,int *uniqueidx,float*uniqueout,int datalen)
{
    //get thrust ptr
    thrust::device_ptr<float> thrust_InOut(dataInOut);
    // thrust::device_ptr<float> thrust_Out(dataOut);
    // create index on dev
//     thrust::device_ptr<int> thrust_Idx(dataIdx);  
//     thrust::device_vector<float> uniqueout(datalen);
//     thrust::device_vector<int> uniqueidx(datalen);
//     //sort
//    // cout<<"length rank"<<datalen<<endl;
//     //thrust::unique
//     thrust::sort_by_key(thrust_InOut,thrust_InOut+datalen,thrust_Idx);
//     auto eend=thrust::unique_copy(thrust_InOut,thrust_InOut+datalen,uniqueout);
//     // for(int i=0;i<datalen;i++)
//     // {
//     //     cout<<"data: "<<thrust_InOut[i]<<" idx: "<<thrust_Idx[i]<<endl;
//     // }
//     auto num_unique = thrust::distance(uniqueout, eend);
    //for test
    //uniqueout.data();
    //uniqueidx.data();

    #ifdef LOG
    // for(int i=0;i<num_unique;i++)
    // {
    // cout<<"datauni: "<<uniqueout[i]<<endl;
    // }
    //getchar();
    #endif


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
    CHECK(cudaStreamCreate(&stream));
    
    CHECK(cudaMallocPitch((void**)&d_ref,&pitchref,ref_width*sizeof(float),ref_height));
    CHECK(cudaMallocPitch((void**)&d_qry,&pitchqry,qry_width*sizeof(float),qry_height));
    
    std::cout<<"pitchref: "<<pitchref<<std::endl;
    std::cout<<"ref_width: "<<ref_width<<std::endl;
   // std::cout<<"pitchqry: "<<pitchqry<<std::endl;

    //cudaMalloc((void**)&d_ref, ref_height * ref_width * sizeof(float));
    //cudaMalloc((void**)&d_qry, qry_height * qry_width * sizeof(float));
    // cudaMalloc((void**)&d_labels, qry_height * ct_num * sizeof(float));
    cudaMalloc((void**)&d_ctids, ctids.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctidx, ctidx.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctdiff, ctdiff.size() * sizeof(uint32));
    // cudaMalloc((void**)&d_ctdidx, ctdidx.size() * sizeof(uint32));
    cout<<"current stream"<<stream<<endl;
    cudaMemcpy2DAsync(d_ref,pitchref, rawdata.ref.data(), ref_width * sizeof(float),ref_width * sizeof(float),ref_height,cudaMemcpyHostToDevice,stream);
    cudaMemcpy2DAsync(d_qry,pitchqry, rawdata.test.data(),qry_width * sizeof(float),qry_width * sizeof(float),qry_height,cudaMemcpyHostToDevice,stream);

    // cudaMemcpyAsync(d_ref, rawdata.ref.data(), ref_height * ref_width * sizeof(float), cudaMemcpyHostToDevice,stream);
    // cudaMemcpyAsync(d_qry, rawdata.test.data(), qry_height * qry_width * sizeof(float), cudaMemcpyHostToDevice,stream);
    // // cudaMemcpy(d_labels, rawdata.labels.data(), qry_height * ct_num * sizeof(float), cudaMemcpyHostToDevice);
    h_labels = rawdata.labels;
    //CHECK( cudaMemcpyAsync(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpy(d_ctids, ctids.data(), ctids.size() * sizeof(uint32), cudaMemcpyHostToDevice));
   
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
    int idx_len=10000;
    cudaMalloc((void**)&d_qry_idx_sample,idx_len*sizeof(int));
    //create origin idx array on CPU and copy to GPU
    h_qry_idx_sample=(int*)malloc(idx_len*sizeof(idx_len));
    for (int i = 0; i < idx_len; ++i)
    {
        h_qry_idx_sample[i] = i;

    }
    CHECK(cudaMemcpyAsync(d_qry_idx_sample,h_qry_idx_sample,idx_len*sizeof(int),cudaMemcpyHostToDevice,stream));
    //use idx sample to reset idx array on GPU
    cudaMalloc((void**)&d_qry_idx,idx_len*sizeof(int));
    cudaMalloc((void**)&d_ref_lines, 100000000 * sizeof(float));
    cudaMalloc((void**)&d_ref_rank, 100000000 * sizeof(float));
    cudaMalloc((void**)&d_score, 100000 * sizeof(float));

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
        float* row_head = (float*)((char*)ref + (uint64)(cell_idx[nx]) * ref_pitch);
        res[nx * gene_len + ny] = row_head[ref_width - gene_idx[ny] - 1];
    }
}

__global__ void rankdata(float* qry, const uint32 len, float* res)
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
    t0=clock();
    
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
    t1=clock();
    time=(float)(t1-t0)/CLOCKS_PER_SEC;
    cout<<"get uniq_genes: "<<time<<endl;
    // cout<<"uniq genes size: "<<uniq_genes.size()<<endl;
    // t0=clock();
    vector<uint32> h_gene_idx(uniq_genes.begin(), uniq_genes.end());
    
    // transfer qry data from cpu to gpu
    CHECK(cudaMemcpy(d_gene_idx, h_gene_idx.data(), h_gene_idx.size()*sizeof(uint32), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qry_idx,d_qry_idx_sample,10000*sizeof(int),cudaMemcpyDeviceToDevice));//copy oringin idx to current idx for sort
  
    // CHECK(cudaMemcpyAsync(d_gene_idx, h_gene_idx.data(), h_gene_idx.size()*sizeof(uint32), cudaMemcpyHostToDevice,stream));
    // CHECK(cudaMemcpyAsync(d_qry_idx,d_qry_idx_sample,10000*sizeof(int),cudaMemcpyDeviceToDevice,stream));//copy oringin idx to current idx for sort
    // CHECK(cudaStreamSynchronize(stream));

  
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
  
    cout<<"thrust sortLen: "<<h_gene_idx.size()<<endl;
    //rank_by_unique(d_qry_line,d_qry_idx,h_gene_idx.size());
   // getchar();
    // thrust::device_ptr<float> thrust_InOut(d_qry_line);
    sort_by_idx1(d_qry_line,d_qry_idx,h_gene_idx.size());
   // sortrank<<< (h_gene_idx.size()/2)/1024 + 1, 1024 >>>(d_qry_line,d_qry_idx,h_gene_idx.size());
    cout<<"thrust end:"<<h_gene_idx.size()<<endl;
    // vector<float> tmp_qryed;
    // vector<int> tmp_qryed_idx;
    // tmp_qryed.resize(h_gene_idx.size(), 0);
    // tmp_qryed_idx.resize(h_gene_idx.size(), 0);

    // CHECK( cudaMemcpy(tmp_qryed_idx.data(), d_qry_idx, h_gene_idx.size()*sizeof(int), cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(tmp_qryed.data(),d_qry_line, h_gene_idx.size()*sizeof(float),cudaMemcpyDeviceToHost));
    // for (int i = 0; i < h_gene_idx.size(); ++i)
    //     {
    //         cout<<"data"<<tmp_qryed[i]<<" ";
    //         cout<<"idx"<<tmp_qryed_idx[i]<<endl;
    //     }
    // getchar();

    // cudaMalloc((void**)&unique_idx,h_gene_idx.size()*sizeof(int));
    // cudaMalloc((void**)&unique_val,h_gene_idx.size()*sizeof(float));
    
    //sort_by_idx1(d_qry_line,d_qry_idx,h_gene_idx.size());
   // sort_by_idx(d_qry_line,d_qry_idx,,,h_gene_idx.size());
    //getchar();
    // thrust::device_ptr<float> thrust_InOut(d_qry_line);
    // thrust::device_ptr<int> thrust_Idx(d_qry_idx);
    // thrust::sort_by_key(dataInOut,dataInOut+h_gene_idx.size(),dataIdx);
    // thrust::device_ptr<float> d_qry_line_Thrust(d_qry_line);

    // t0=clock();

    // thrust::sort(d_qry_line_Thrust,d_qry_line_Thrust+h_gene_idx.size());
    time=(float)(t1-t0)/CLOCKS_PER_SEC;
    cout<<"sort : "<<time<<endl;

    //thrust::sort_by_key(d_qry_line_Thrust,d_qry_line_Thrust+h_gene_idx.size(),ids);
    cout<<"thrust end"<<endl;
  //  getchar();
    
   // rankdata<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_qry_line, h_gene_idx.size(), d_qry_rank);
   
    // t1=clock();
    // time=(float)(t1-t0)/CLOCKS_PER_SEC;
    // cout<<"rankdata 1st : "<<time<<endl;
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
        t0=clock();
        get_device_ref_lines<<< gridDim, blockDim >>>
            (d_gene_idx, h_gene_idx.size(), d_ctids+pos, len, d_ref, ref_width, pitchref, d_ref_lines);
        t1=clock();
        time=(float)(t1-t0)/CLOCKS_PER_SEC;
        cout<<"label:"<<label<<" get_device_ref_lines : "<<time<<endl;
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
        t0=clock();
        for (int i = 0; i < len; ++i)
        {
            //sortrank<<< (h_gene_idx.size()/2)/1024 + 1, 1024 >>>(d_qry_line,d_qry_idx,h_gene_idx.size());
          //  sortrank<<<(h_gene_idx.size()/2)/1024 + 1, 1024>>>(d_ref_lines+i*h_gene_idx.size(),d_qry_idx,h_gene_idx.size());
    //         CHECK( cudaMemcpy(tmp_qryed.data(), d_ref_lines+i*h_gene_idx.size(), h_gene_idx.size()*sizeof(float), cudaMemcpyDeviceToHost));
    //         CHECK( cudaMemcpy(tmp_qryed_idx.data(), d_qry_idx, h_gene_idx.size()*sizeof(int), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < h_gene_idx.size(); ++i)
    //     {
    //         cout<<"data"<<tmp_qryed[i]<<" ";
    //         cout<<"idx"<<tmp_qryed_idx[i]<<endl;
    //     }
    // getchar();
        sort_by_idx1(d_ref_lines+i*h_gene_idx.size(),d_qry_idx,h_gene_idx.size());
           // rankdata<<< h_gene_idx.size()/1024 + 1, 1024 >>>(d_ref_lines+i*h_gene_idx.size(), h_gene_idx.size(), d_ref_rank+i*h_gene_idx.size());
        }
        t1=clock();
        time=(float)(t1-t0)/CLOCKS_PER_SEC;
        cout<<"label:"<<label<<" rankdata  len "<<len<<"time:"<<time<<endl;
        // vector<float> tmp_ref_line;
        // tmp_ref_line.resize(h_gene_idx.size()*len, 0);
        // cudaMemcpy(tmp_ref_line.data(), d_ref_lines, h_gene_idx.size()*len*sizeof(float), cudaMemcpyDeviceToHost);
        // float max_val = 0, total_val = 0;
        // for (int i = 0; i < tmp_ref_line.size(); ++i)
        // {
        //     max_val = max(max_val, tmp_ref_line[i]);
        //     total_val += tmp_ref_line[i];
        // }
        // cout<<max_val<<" "<<total_val<<endl;

        // spearman
        // cudaMemset(d_score, 0, 1000 * sizeof(float));
        t0=clock();
        spearman<<< len/1024 + 1, 1024 >>>(d_qry_rank, d_ref_rank, h_gene_idx.size(), len, d_score);
        t1=clock();
        time=(float)(t1-t0)/CLOCKS_PER_SEC;
        cout<<"label:"<<label<<" spearman : "<<time<<endl;
        vector<float> h_score;
        h_score.resize(len, 0);
        cudaMemcpy(h_score.data(), d_score, len*sizeof(float), cudaMemcpyDeviceToHost);
        // cout<<"score len: "<<len<<endl;
        // if (scores.size() == 1)
        // {
        //     auto ele = std::minmax_element(h_score.begin(), h_score.end());
        //     cout<<"ele: "<<*ele.first<<" "<<*ele.second<<endl;
        // }
        float score = percentile(h_score, len, 0.8);
        // cout<<"score: "<<score<<endl;
        scores.push_back(score);
        
        // cudaMemset(d_ref_lines, 0, h_gene_idx.size() * len * sizeof(float));
    }

    // for (auto& score : scores)
    //     cout<<score<<" ";
    // cout<<endl;
    t0=clock();
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
    t1=clock();
    time=(float)(t1-t0)/CLOCKS_PER_SEC;
    cout<<"rest etc : "<<time<<endl;
    return res;
}

vector<uint32> finetune()
{
    // process each cell
    vector<uint32> res;
    cout<<"cell num:"<<qry_height<<endl;
    clock_t startT,endT;
    float timecell;
    // for (int i = 0; i < 1; ++i)
    //for (int i = 26; i < 27; ++i)
    for (int i = 0; i < qry_height; ++i)
    {
        startT=clock();
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
            cout<<"top_labels size"<<top_labels.size()<<endl;
            top_labels = finetune_round(qry_head, top_labels);
            // for (auto& label : top_labels)
            //     cout<<label<<endl;
        }
        res.push_back(top_labels.front());
        if (i % 10 == 0)
        {
            auto now = std::chrono::system_clock::now();
            std::time_t curr_time = std::chrono::system_clock::to_time_t(now);
            cout<<std::ctime(&curr_time)<<"\tprocessed "<<i<<" cells"<<endl;
        }
        endT=clock();
        timecell=(float)(endT-startT)/CLOCKS_PER_SEC;
        cout<<"cell"<<i<<"procTime:"<<timecell<<endl;
    }
 
    return res;
}
