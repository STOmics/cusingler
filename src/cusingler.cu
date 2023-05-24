#include "cusingler.cuh"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <thread>

#include "cuda_runtime.h"
#include "math_constants.h"

cudaStream_t   stream;
uint16 *       d_ref, *d_qry;
vector<float>  h_labels;
uint32         ref_height, ref_width, qry_height, qry_width;
uint32         ct_num;
vector<uint32> h_ctidx;
vector<uint32> h_ctdiff, h_ctdidx;
size_t         pitchref;
size_t         pitchqry;

uint32 *d_gene_idx, *d_cell_idx;
uint16 *d_ref_lines, *d_qry_line;
float * d_ref_rank, *d_qry_rank;
float*  d_score;

int shared_mem_per_block;

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

// unit is MB
uint32 getUsedMem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (total - free) / 1024 / 1024;
}

uint32 getFreeMem()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free / 1024 / 1024;
}

bool initGPU(const int gpuid)
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    if (gpuid < devicesCount)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, gpuid);
        if (deviceProperties.major >= 2 && deviceProperties.minor >= 0)
        {
            shared_mem_per_block = deviceProperties.sharedMemPerBlock;
            cudaSetDevice(gpuid);
            return true;
        }
    }
    return false;
}

bool destroy()
{
    cudaFree(d_ref);
    cudaFree(d_qry);

    cudaFree(d_gene_idx);
    cudaFree(d_cell_idx);
    cudaFree(d_qry_line);
    cudaFree(d_ref_lines);
    cudaFree(d_qry_rank);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    cudaStreamDestroy(stream);

    return true;
}

bool destroy_score()
{
    cudaFree(d_ref);
    cudaFree(d_qry);

    cudaFree(d_gene_idx);
    cudaFree(d_cell_idx);
    cudaFree(d_qry_line);

    cudaFree(d_qry_rank);
    cudaFree(d_ref_rank);
    cudaFree(d_score);

    return true;
}

__global__ void rankdata_pitch(uint16* dataIn, float* dataOut, const int datalen,
                               const size_t pitch, const int batch)
{  // no shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen * batch)
    {
        int     batchid      = tid / datalen;
        float   equl_cnt     = 0;
        float   less_cnt     = 1;
        uint16* dataIn_batch = ( uint16* )(( char* )dataIn + batchid * pitch);
        for (int i = 0; i < datalen; i++)
        {

            if (dataIn_batch[tid] > dataIn_batch[i])
            {
                less_cnt++;
            }
            else if (dataIn_batch[tid] == dataIn_batch[i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + (equl_cnt - 1) / 2;
    }
}

// for qry 1 line rank
__global__ void rankdata(uint16* dataIn, float* dataOut, const int datalen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen)
    {
        float equl_cnt = 0;
        float less_cnt = 1;
        for (int i = 0; i < datalen; i++)
        {
            if (dataIn[tid] > dataIn[i])
            {
                less_cnt++;
            }
            else if (dataIn[tid] == dataIn[i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + (equl_cnt - 1) / 2;
    }
}

__global__ void rankdata(float* dataIn, float* dataOut, const int datalen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen)
    {
        float equl_cnt = 0;
        float less_cnt = 1;
        for (int i = 0; i < datalen; i++)
        {
            if (dataIn[tid] > dataIn[i])
            {
                less_cnt++;
            }
            else if (dataIn[tid] == dataIn[i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + (equl_cnt - 1) / 2;
    }
}

bool copyin_score(InputData& rawdata)
{
    ref_height = rawdata.ref_height;
    ref_width  = rawdata.ref_width;
    qry_height = rawdata.qry_height;
    qry_width  = rawdata.qry_width;
    ct_num     = rawdata.ct_num;

    size_t estimated_mem = 0;
    estimated_mem += (size_t(ref_height)*ref_width+size_t(qry_height)*qry_width)*(sizeof(uint16)+sizeof(float));
    estimated_mem += qry_width*(sizeof(uint16)+sizeof(uint32));
    estimated_mem += ref_height*(sizeof(uint32)+1024*sizeof(float));
    estimated_mem /= 1024*1024;
    estimated_mem += 255;   // system memory
    auto free_mem = getFreeMem();
    if ((estimated_mem+500) > free_mem)
    {
        cerr<<"Need gpu memory(MB): "<<estimated_mem+500<<" less than free memory(MB): "<<free_mem<<endl;
        return false;
    }

    CHECK(cudaMalloc(( void** )&d_ref, rawdata.ref.size() * sizeof(uint16)));
    CHECK(cudaMemcpy(d_ref, rawdata.ref.data(), rawdata.ref.size() * sizeof(uint16),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(( void** )&d_qry, rawdata.qry.size() * sizeof(uint16)));
    CHECK(cudaMemcpy(d_qry, rawdata.qry.data(), rawdata.qry.size() * sizeof(uint16),
                     cudaMemcpyHostToDevice));

    h_ctidx = rawdata.ctidx;

    CHECK(cudaMalloc(( void** )&d_gene_idx, qry_width * sizeof(uint32)));
    CHECK(cudaMalloc(( void** )&d_cell_idx, ref_height * sizeof(uint32)));
    CHECK(cudaMalloc(( void** )&d_qry_line, qry_width * sizeof(uint16)));

    CHECK(cudaMalloc(( void** )&d_ref_rank, rawdata.ref.size() * sizeof(float)));
    CHECK(cudaMalloc(( void** )&d_qry_rank, rawdata.qry.size() * sizeof(float)));
    CHECK(cudaMalloc(( void** )&d_score, 1024 * ref_height * sizeof(float)));

    std::cout << "score() used gpu mem(MB): " << estimated_mem << std::endl;

    return true;
}

bool copyin(InputData& rawdata)
{
    ref_height = rawdata.ref_height;
    ref_width  = rawdata.ref_width;
    qry_height = rawdata.qry_height;
    qry_width  = rawdata.qry_width;
    ct_num     = rawdata.ct_num;

    size_t estimated_mem = 0;
    estimated_mem += (size_t(ref_height)*ref_width+size_t(qry_height)*qry_width)*sizeof(uint16);
    estimated_mem += qry_width*(sizeof(uint16)+sizeof(uint32)+sizeof(float));
    estimated_mem += ref_height*(sizeof(uint32)+sizeof(float));
    estimated_mem += size_t(ref_height)*ref_width*(sizeof(uint16)+sizeof(float));
    estimated_mem /= 1024*1024;
    estimated_mem += 255;   // system memory
    auto free_mem = getFreeMem();
    if ((estimated_mem+500) > free_mem)
    {
        cerr<<"Need gpu memory(MB): "<<estimated_mem+500<<" less than free memory(MB): "<<free_mem<<endl;
        return false;
    }

    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMallocPitch(( void** )&d_ref, &pitchref, ref_width * sizeof(uint16),
                          ref_height));
    CHECK(cudaMallocPitch(( void** )&d_qry, &pitchqry, qry_width * sizeof(uint16),
                          qry_height));

    CHECK(cudaMemcpy2DAsync(d_ref, pitchref, rawdata.ref.data(), ref_width * sizeof(uint16),
                            ref_width * sizeof(uint16), ref_height,
                            cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpy2DAsync(d_qry, pitchqry, rawdata.qry.data(), qry_width * sizeof(uint16),
                            qry_width * sizeof(uint16), qry_height,
                            cudaMemcpyHostToDevice, stream));

    h_labels = rawdata.labels;

    h_ctidx = rawdata.ctidx;

    h_ctdiff = rawdata.ctdiff;
    h_ctdidx = rawdata.ctdidx;

    cudaStreamSynchronize(stream);

    CHECK(cudaMalloc(( void** )&d_gene_idx, qry_width * sizeof(uint32)));
    CHECK(cudaMalloc(( void** )&d_cell_idx, ref_height * sizeof(uint32)));
    CHECK(cudaMalloc(( void** )&d_qry_line, qry_width * sizeof(uint16)));
    CHECK(cudaMalloc(( void** )&d_qry_rank, qry_width * sizeof(float)));

    CHECK(cudaMalloc(( void** )&d_ref_lines, rawdata.ref.size() * sizeof(uint16)));
    CHECK(cudaMalloc(( void** )&d_ref_rank, rawdata.ref.size() * sizeof(float)));
    CHECK(cudaMalloc(( void** )&d_score, ref_height * sizeof(float)));

    std::cout << "finetune() used gpu mem(MB): " << estimated_mem << std::endl;

    return true;
}

__global__ void get_device_qry_line(uint32* gene_idx, uint16* qry, const uint32 len,
                                    const uint32 gene_len, uint16* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len)
    {
        res[tid] = qry[gene_idx[tid]];
    }
}

__global__ void get_device_ref_pitch(uint32* gene_idx, const uint32 gene_len,
                                     uint32* cell_idx, const uint32 cell_len, uint16* ref,
                                     const uint32 ref_width, const size_t ref_pitch,
                                     const uint32 ref_lines_width,
                                     const size_t ref_lines_pitch, uint16* ref_lines)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < cell_len && ny < gene_len)
    {
        uint16* row_head =
            ( uint16* )(( char* )ref + ( uint64 )(cell_idx[nx]) * ref_pitch);
        uint16* row_head_lines =
            ( uint16* )(( char* )ref_lines + ( uint64 )( nx )*ref_lines_pitch);
        row_head_lines[ny] = row_head[ref_width - gene_idx[ny] - 1];
    }
}
__global__ void get_device_ref_lines(const uint32* gene_idx, const uint32 gene_len,
                                     const uint32* cell_idx, const uint32 cell_len,
                                     uint16* ref, const uint32 ref_width,
                                     const uint64 ref_pitch, uint16* res)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (ny < gene_len)
    {
        uint16* row_head        = ( uint16* )(( char* )ref + cell_idx[nx] * ref_pitch);
        res[nx * gene_len + ny] = row_head[gene_idx[ny]];
    }
}
__global__ void get_device_ref_lines(uint32* gene_idx, const uint32 gene_len,
                                     uint32* cell_idx, const uint32 cell_len, float* ref,
                                     const uint32 ref_width, const uint64 ref_pitch,
                                     float* res)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < cell_len && ny < gene_len)
    {
        float* row_head = ( float* )(( char* )ref + ( uint64 )(cell_idx[nx]) * ref_pitch);
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
        res[tid] = r + float(s - 1) / 2;
    }
}
/*
    datain : ref
    dataout: ref_rank
    datalen: gene_len
    batch: cell_len
*/
__global__ void rankdata_batch(uint16* dataIn, float* dataOut, const int datalen,
                               const int batch)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < datalen * batch)
    {
        int batchid  = tid / datalen;
        int equl_cnt = 0;
        int less_cnt = 1;
        for (int i = 0; i < datalen; i++)
        {

            if (dataIn[tid] > dataIn[batchid * datalen + i])
            {
                less_cnt++;
            }
            else if (dataIn[tid] == dataIn[batchid * datalen + i])
            {
                equl_cnt++;
            }
        }
        dataOut[tid] = less_cnt + ( float )(equl_cnt - 1) / 2;
    }
}

// basic rankdata method using bincount
__global__ void rankdata_bin(uint16* qry, const uint32 cols, const uint32 rows,
                             float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows)
    {
        uint16* q = qry + tid * cols;
        float*  r = res + tid * cols;
        // travel for getting bin count
        uint16 bins[80] = { 0 };
        for (int i = 0; i < cols; ++i)
            bins[q[i]]++;

        // calculate real rank
        float ranks[80];
        float start = 0;
        for (int i = 0; i < 80; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[i] = start + (bins[i] + 1) * 0.5;
            start += bins[i];
        }

        // assign rank value
        for (int i = 0; i < cols; ++i)
            r[i] = ranks[q[i]];
    }
}

// using shared memory
__global__ void rankdata_bin2(uint16* qry, const uint32 cols, const uint32 rows,
                              float* res)
{
    __shared__ uint16 bins[80 * 64];
    __shared__ float  ranks[80 * 64];
    int               bid = threadIdx.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows)
    {
        for (int i = 0; i < 80; ++i)
        {
            bins[bid * 80 + i] = 0;
        }

        uint16* q = qry + tid * cols;
        float*  r = res + tid * cols;
        // travel for getting bin count
        for (int i = 0; i < cols; ++i)
            bins[bid * 80 + q[i]]++;

        // calculate real rank
        float start = 0;
        for (int i = 0; i < 80; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[bid * 80 + i] = start + (bins[bid * 80 + i] + 1) * 0.5;
            start += bins[bid * 80 + i];
        }

        // assign rank value
        for (int i = 0; i < cols; ++i)
            r[i] = ranks[bid * 80 + q[i]];
    }
}

// one block threads for each cell
__global__ void rankdata_bin3_pitch(uint16* qry, const size_t pitch, const uint32 cols,
                                    const uint32 rows, float* res)
{
    __shared__ int   bins[128];
    __shared__ float ranks[128];
    int              bid = blockIdx.x;
    int              tid = threadIdx.x;

    int step = cols / 64;
    if (cols % 64 != 0)
        step++;

    if (bid < rows)
    {
        bins[tid * 2]     = 0;
        bins[tid * 2 + 1] = 0;
        __syncthreads();
        uint16* q = ( uint16* )(( char* )qry + bid * pitch);
        // uint16* q = qry + bid*cols;
        float* r = res + bid * cols;
        // travel for getting bin count
        for (int i = tid * step; i < (tid + 1) * step; ++i)
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
                ranks[i] = start + (bins[i] + 1) * 0.5;
                start += bins[i];
            }
        }
        __syncthreads();

        // assign rank value
        for (int i = tid * step; i < (tid + 1) * step; ++i)
        {
            if (i < cols)
                r[i] = ranks[q[i]];
        }
    }
}

__global__ void rankdata_bin3(uint16* qry, const uint32 cols, const uint32 rows,
                              const uint64 max_uniq_gene, float* res)
{
    extern __shared__ int mem[];
    int*                  bins  = mem;
    float*                ranks = ( float* )&bins[max_uniq_gene];

    int bid     = blockIdx.x;
    int tid     = threadIdx.x;
    int threads = blockDim.x;

    int step = cols / threads;
    if (cols % threads != 0)
        step++;
    int step1 = max_uniq_gene / threads;
    if (max_uniq_gene % threads != 0)
        step1++;

    for (int i = 0; i < step1; ++i)
    {
        if ((tid * step1 + i) < max_uniq_gene)
        {
            bins[tid * step1 + i] = 0;
        }
    }
    __syncthreads();

    uint16* q = qry + bid * cols;
    float*  r = res + bid * cols;
    // travel for getting bin count
    for (int i = tid * step; i < (tid + 1) * step; ++i)
    {
        if (i < cols)
            atomicAdd(&bins[q[i]], 1);
    }
    __syncthreads();

    // calculate real rank
    if (tid == 0)
    {
        int start = 0;
        for (int i = 0; i < max_uniq_gene; ++i)
        {
            // if (bins[i] == 0) continue;
            ranks[i] = start + (bins[i] + 1) * 0.5;
            start += bins[i];
        }
    }
    __syncthreads();

    // assign rank value
    for (int i = tid * step; i < (tid + 1) * step; ++i)
    {
        if (i < cols)
            r[i] = ranks[q[i]];
    }
}

// 1xN vs MxN -> MX1
__global__ void spearman_reduce(float* qry, float* ref, const uint32 gene_num,
                                const uint32 cell_num, float* score)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float mean = (gene_num + 1) / 2.0;
    int   step = gene_num / 128;
    if (gene_num % 128 != 0)
        step++;

    __shared__ float sumxy[128];
    __shared__ float sumxx[128];
    __shared__ float sumyy[128];

    sumxy[tid] = 0;
    sumxx[tid] = 0;
    sumyy[tid] = 0;

    for (int i = tid * step; i < (tid + 1) * step; ++i)
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

// KxN vs MxN -> MxK
__global__ void spearman(float* qry, float* ref, const uint32 gene_num,
                         const uint32 qry_cell_num, const uint32 ref_cell_num,
                         float* score)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float mean = (gene_num + 1) / 2.0;
    int   step = gene_num / 128;
    if (gene_num % 128 != 0)
        step++;

    __shared__ float sumxy[128];
    __shared__ float sumxx[128];
    __shared__ float sumyy[128];

    for (int k = 0; k < qry_cell_num; ++k)
    {
        sumxy[tid] = 0;
        sumxx[tid] = 0;
        sumyy[tid] = 0;
        __syncthreads();

        for (int i = tid * step; i < (tid + 1) * step; ++i)
        {
            if (i < gene_num)
            {
                float x = qry[k * gene_num + i] - mean;
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
                score[k * ref_cell_num + bid] = sumxy[0] / divisor;
            else
                score[k * ref_cell_num + bid] = CUDART_NAN_F;
        }
        __syncthreads();
    }
}

__global__ void spearman(float* qry, const uint32 start, float* ref,
                         const uint32 gene_num, const uint32 qry_cell_num,
                         const uint32 ref_cell_num, float* score)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (start + tid >= qry_cell_num)
        return;

    float mean = (gene_num + 1) / 2.0;

    float sumxy = 0, sumxx = 0, sumyy = 0;

    for (int i = 0; i < gene_num; ++i)
    {
        float x = qry[(start + tid) * gene_num + i] - mean;
        float y = ref[bid * gene_num + i] - mean;
        sumxy += x * y;
        sumxx += x * x;
        sumyy += y * y;
    }

    float divisor = sqrt(sumxx * sumyy);
    if (divisor != 0)
        score[tid * ref_cell_num + bid] = sumxy / divisor;
    else
        score[tid * ref_cell_num + bid] = CUDART_NAN_F;
}

float percentile(vector<float> arr, int len, float p)
{
    if (len <= 1)
        return arr.front();

    float res;
    std::sort(arr.begin(), arr.begin() + len);

    vector<float> index;
    float         step = 1.0 / (len - 1);
    for (int i = 0; i < len; ++i)
        index.push_back(i * step);

    if (p <= index.front())
    {
        res = arr[0];
    }
    else if (index.back() <= p)
    {
        res = arr[len - 1];
    }
    else
    {
        auto  it        = lower_bound(index.begin(), index.end(), p);
        float prevIndex = *(it - 1);
        float prevValue = arr.at(it - index.begin() - 1);
        float nextValue = arr.at(it - index.begin());
        // linear interpolation
        res = (p - prevIndex) * (nextValue - prevValue) / step + prevValue;
    }
    return res;
}

inline bool bincount(const uint64 max_uniq_gene)
{
    // TODO: Get total amount of shared memory per block from device
    return max_uniq_gene <= min(1024, shared_mem_per_block/8);
}

vector<int> get_label(InputData& rawdata, const uint64 max_uniq_gene, int cores)
{
    if (bincount(max_uniq_gene))
    {
        // Using bincount method
        int shared_mem = max_uniq_gene * (sizeof(int) + sizeof(float));
        rankdata_bin3<<<ref_height, 64, shared_mem>>>(d_ref, ref_width, ref_height,
                                                      max_uniq_gene, d_ref_rank);
        CHECK(cudaGetLastError());

        rankdata_bin3<<<qry_height, 64, shared_mem>>>(d_qry, qry_width, qry_height,
                                                      max_uniq_gene, d_qry_rank);
        CHECK(cudaGetLastError());
    }
    else
    {
        rankdata_batch<<<(ref_height * ref_width - 1) / 512 + 1, 512>>>(
            d_ref, d_ref_rank, ref_width, ref_height);
        CHECK(cudaGetLastError());

        rankdata_batch<<<(qry_height * qry_width - 1) / 512 + 1, 512>>>(
            d_qry, d_qry_rank, qry_width, qry_height);
        CHECK(cudaGetLastError());
    }

    // get all qry rank and calculate score
    vector<int> first_labels(rawdata.labels.size() / rawdata.ct_num, 0);
    auto task = [&](vector<float>& score, int thread_id, int width, vector<uint32>& idx,
                    int ct_num, int step_size, vector<int>& res)
    {
        int height = score.size() / width;
        for (int i = 0; i < step_size; ++i)
        {
            int line = thread_id * step_size + i;
            if (line >= height)
                break;
            int           start     = line * width;
            int           total_len = start;
            vector<float> scores;
            for (int j = 0; j < ct_num; ++j)
            {
                size_t len = idx[j * 2 + 1];
                total_len += len;

                vector<float> tmp(score.begin() + start, score.begin() + total_len);
                float         score = percentile(tmp, len, 0.8);
                scores.push_back(score);
                start += len;
            }
            auto ele           = std::minmax_element(scores.begin(), scores.end());
            first_labels[line] = (ele.second - scores.begin());
            float thre         = *ele.second - 0.05;  // max-0.05
            for (int i = 0; i < scores.size(); ++i)
            {
                if (scores[i] >= thre)
                {
                    res[line * ct_num + i] = 1;
                }
                else
                {
                    res[line * ct_num + i] = 0;
                }
            }
        }
    };

    // Get suitable thread number
    while (1024 % cores != 0)
        cores--;
    cores = min(cores, 8);
    for (int line = 0; line < qry_height; line += 1024)
    {
        spearman<<<ref_height, 1024>>>(d_qry_rank, line, d_ref_rank, qry_width,
                                       qry_height, ref_height, d_score);
        CHECK(cudaGetLastError());

        vector<float> h_score;
        h_score.resize(1024 * ref_height, 0);

        CHECK(cudaMemcpy(h_score.data(), d_score, 1024 * ref_height * sizeof(float),
                         cudaMemcpyDeviceToHost));

        vector<int> h_labels;
        h_labels.resize(1024 * ct_num, 0);

        vector<thread> threads;
        for (int i = 0; i < cores; ++i)
        {
            thread th(task, std::ref(h_score), i, ref_height, std::ref(h_ctidx), ct_num, 1024 / cores,
                      std::ref(h_labels));
            threads.push_back(std::move(th));
        }
        for (auto& th : threads)
        {
            th.join();
        }

        for (int i = 0; i < h_labels.size(); ++i)
            if ((line * ct_num + i) < qry_height * ct_num)
                rawdata.labels[line * ct_num + i] = h_labels[i];
    }

    return first_labels;
}

vector<uint32> finetune_round(uint16* qry, vector<uint32> top_labels,
                              const uint64 max_uniq_gene)
{
    set<uint32> uniq_genes;
    int         gene_thre = round(500 * pow((2 / 3.0), log2(top_labels.size())));

    for (auto& i : top_labels)
    {
        for (auto& j : top_labels)
        {
            if (i == j)
                continue;
            int pos = h_ctdidx[(i * ct_num + j) * 2];
            int len = h_ctdidx[(i * ct_num + j) * 2 + 1];
            if (len > gene_thre)
                len = gene_thre;
            uniq_genes.insert(h_ctdiff.begin() + pos, h_ctdiff.begin() + pos + len);
        }
    }

    vector<uint32> h_gene_idx(uniq_genes.begin(), uniq_genes.end());

    // transfer qry data from cpu to gpu
    CHECK(cudaMemcpy(d_gene_idx, h_gene_idx.data(), h_gene_idx.size() * sizeof(uint32),
                     cudaMemcpyHostToDevice));

    get_device_qry_line<<<(h_gene_idx.size() - 1) / 1024 + 1, 1024>>>(
        d_gene_idx, qry, h_gene_idx.size(), qry_width, d_qry_line);
    CHECK(cudaGetLastError());

    // get rank of qry data
    rankdata<<<(h_gene_idx.size() - 1) / 1024 + 1, 1024>>>(d_qry_line, d_qry_rank,
                                                           h_gene_idx.size());
    CHECK(cudaGetLastError());

    vector<pair<size_t, size_t>> temp;
    size_t                       total_len = 0;
    for (auto& label : top_labels)
    {
        uint32 pos = h_ctidx[label * 2];
        uint32 len = h_ctidx[label * 2 + 1];
        if (temp.empty() || (temp.back().first + temp.back().second) != pos)
        {
            temp.push_back({ pos, len });
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
    for (auto& tmp : temp)
    {
        size_t pos = tmp.first;
        size_t len = tmp.second;
        std::iota(h_cell_idx.begin() + total_len, h_cell_idx.begin() + total_len + len,
                  pos);
        total_len += len;
    }

    CHECK(cudaMemcpy(d_cell_idx, h_cell_idx.data(), h_cell_idx.size() * sizeof(uint32),
                     cudaMemcpyHostToDevice));

    dim3 blockDim(1, 512);
    dim3 gridDim(h_cell_idx.size(), h_gene_idx.size() / 512 + 1);

    get_device_ref_lines<<<gridDim, blockDim>>>(d_gene_idx, h_gene_idx.size(), d_cell_idx,
                                                h_cell_idx.size(), d_ref, ref_width,
                                                ( uint64 )pitchref, d_ref_lines);
    CHECK(cudaGetLastError());

    if (bincount(max_uniq_gene))
    {
        int shared_mem = max_uniq_gene * (sizeof(int) + sizeof(float));
        rankdata_bin3<<<total_len, 64, shared_mem>>>(
            d_ref_lines, h_gene_idx.size(), total_len, max_uniq_gene, d_ref_rank);
        CHECK(cudaGetLastError());
    }
    else
    {
        // len=h_cell_idx.size()
        rankdata_batch<<<(h_cell_idx.size() * h_gene_idx.size() - 1) / 512 + 1, 512>>>(
            d_ref_lines, d_ref_rank, h_gene_idx.size(), h_cell_idx.size());
        CHECK(cudaGetLastError());
    }

    // spearman
    spearman_reduce<<<total_len, 128>>>(d_qry_rank, d_ref_rank, h_gene_idx.size(),
                                        total_len, d_score);
    CHECK(cudaGetLastError());

    vector<float> h_score;
    h_score.resize(total_len, 0);

    CHECK(cudaMemcpy(h_score.data(), d_score, total_len * sizeof(float),
                     cudaMemcpyDeviceToHost));
    uint32 start = 0;
    total_len    = 0;
    vector<float> scores;
    for (auto& label : top_labels)
    {
        uint32 len = h_ctidx[label * 2 + 1];
        total_len += len;

        vector<float> tmp(h_score.begin() + start, h_score.begin() + total_len);
        float         score = percentile(tmp, len, 0.8);
        scores.push_back(score);
        start += len;
    }

    auto  ele  = std::minmax_element(scores.begin(), scores.end());
    float thre = *ele.second - 0.05;
    if (std::isnan(*ele.second))
    {
        cerr << "Got score 'nan'" << endl;
        exit(-1);
    }
    vector<uint32> res;
    for (uint32 i = 0; i < scores.size(); ++i)
    {
        if (scores[i] <= *ele.first || scores[i] < thre)
            continue;
        else
            res.push_back(top_labels[i]);
    }
    if (res.empty())
        res.push_back(top_labels.front());

    return res;
}
vector<uint32> cufinetune(const uint64 max_uniq_gene)
{
    Timer timer("s");

    vector<uint32> res;
    // process each cell
    for (int i = 0; i < qry_height; ++i)
    {
        uint16* qry_head = ( uint16* )(( char* )d_qry + i * pitchqry);

        vector<uint32> top_labels;

        uint32 start = i * ct_num;
        for (int pos = 0; pos < ct_num; ++pos)
        {
            if (h_labels.at(start + pos) != 0)
                top_labels.push_back(pos);
        }

        while (top_labels.size() > 1)
        {
            top_labels = finetune_round(qry_head, top_labels, max_uniq_gene);
        }

        res.push_back(top_labels.front());
        if (i != 0 && i % 1000 == 0)
        {
            cout << "processed " << i << " cells cost time(s): " << timer.toc() << endl;
        }
    }

    return res;
}
