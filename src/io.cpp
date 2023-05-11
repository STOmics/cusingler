/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "io.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <cassert>
using namespace std;

#include "H5Cpp.h"
using namespace H5;

bool readLabels(string filename, InputData& data)
{
    // open h5 file handle
    H5File* file = new H5File(filename.c_str(), H5F_ACC_RDONLY);

    auto& labels = data.labels;

    DataSet   dataset   = DataSet(file->openDataSet("/tmp"));
    auto      datatype  = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    int       rank      = dataspace.getSimpleExtentNdims();
    hsize_t   dims[rank];
    dataspace.getSimpleExtentDims(dims, NULL);

    size_t test_size = 1;
    for (int i = 0; i < rank; ++i)
        test_size *= dims[i];
    labels.resize(test_size);
    dataset.read(&labels[0], datatype);

    cout << "labels size: " << labels.size() << " " << dims[0] << " x " << dims[1]
         << endl;

    // clear resources
    delete file;

    return true;
}

template<typename T>
vector<T> getDataset(Group& group, string name)
{
    auto   dataset   = DataSet(group.openDataSet(name.c_str()));
    auto      datatype  = dataset.getDataType();
    auto dataspace = dataset.getSpace();
    int       rank      = dataspace.getSimpleExtentNdims();
    hsize_t   dims[rank];
    dataspace.getSimpleExtentDims(dims, NULL);

    uint64 size = dims[0];
    vector<T> data;
    data.resize(size);
    dataset.read(data.data(), datatype);

    // cout<<"data size: "<<data.size()<<endl;
    return data;
}

bool DataParser::loadRefMatrix()
{
    // Open h5 file handle
    H5File* file = new H5File(ref_file.c_str(), H5F_ACC_RDONLY);

    // Load matrix data and shape
    {
        auto group(file->openGroup("/X"));

        Attribute attr(group.openAttribute("shape"));
        auto datatype  = attr.getDataType();
        vector<uint64> shapes(2, 0);
        attr.read(datatype, shapes.data());
        ref_height = shapes[0];
        ref_width = shapes[1];
        cout<<"Ref shape: "<<ref_height<<" x "<<ref_width<<endl;

        ref_data = getDataset<float>(group, "data");
        ref_indices = getDataset<int>(group, "indices");
        ref_indptr = getDataset<int>(group, "indptr");

        set<float> m(ref_data.begin(), ref_data.end());
        cout<<"ref data uniq elements: "<<m.size()<<endl;
    }

    // Load celltypes of per cell
    {
        auto group(file->openGroup("/obs/ClusterName"));

        celltype_codes = getDataset<uint8>(group, "codes");
        uniq_celltypes = getDataset<char*>(group, "categories");
        label_num = uniq_celltypes.size();
    }

    // clear resources
    delete file;

    return true;
}

bool DataParser::trainData()
{
    Timer timer("ms");
    // Groupby celltype of each cell
    map<uint8, vector<uint32>> group_by;
    for (uint32 i = 0; i < celltype_codes.size(); ++i)
    {
        group_by[celltype_codes[i]].push_back(i);
    }

    // cout<<"groupby time: "<<timer.toc()<<endl;

    // Calculate median gene value for each celltype
    map<uint8, vector<float>> median_map;
    for (auto& [ct, idxs] : group_by)
    {
        vector<float> sub_ref;
        sub_ref.resize(idxs.size() * ref_width, 0);
        int line = 0;
        for (auto& idx : idxs)
        {
            auto start = ref_indptr[idx];
            auto end = ref_indptr[idx+1];
            for (uint32 i = start; i < end; ++i)
            {
                sub_ref[line*ref_width + ref_indices[i]] = ref_data[i];
            }
            line++;
        }
        // cout<<"sub ref time: "<<timer.toc()<<endl;

        vector<float> median;
        for (uint32 i = 0; i < ref_width; ++i)
        {
            vector<float> cols;
            for (uint32 j = 0; j < idxs.size(); ++j)
            {
                cols.push_back(sub_ref[j*ref_width+i]);
            }
            if (cols.size() % 2 == 0)
            {
                std::sort(cols.begin(), cols.end());
                median.push_back((cols[cols.size()/2]+cols[cols.size()/2-1])/2);
            }
            else
            {
                std::nth_element(cols.begin(), cols.begin()+cols.size()/2, cols.end());
                median.push_back(cols[cols.size()/2]);
            }
        }
        median_map.insert({ct, median});
        // cout<<"median time: "<<timer.toc()<<endl;
    }
    cout<<"median time: "<<timer.toc()<<endl;

    // Calculate difference for each two celltypes
    uint32 idx_start = 0;
    size_t gene_thre = round(500 * pow((2 / 3.0), log2(uniq_celltypes.size())));

    // set<uint32> exclude_genes{2034, 2599, 6326, 7525, 7637, 8125, 9321, 10246, 13544, 17496};
    for (auto& [k1, v1] : median_map)
    {
        for (auto& [k2, v2] : median_map)
        {
            // if (k1 != 6 || k2 != 17)
            //     continue;

            if (k1 == k2)
            {
                // padding zero
                ref_train_idxs.push_back(0);
                ref_train_idxs.push_back(0);
                continue;
            }
            // Get diff of two array
            vector<pair<float, uint32>> diff;
            for (int i = 0; i < ref_width; ++i)
            {
                diff.push_back({v1[i] - v2[i], ref_width-i-1});
            }
            
            // Sort by ascending order
            std::sort(diff.begin(),diff.end(),std::greater<pair<float,uint32>>());
                
            // Only need the score > 0
            int i = 0;
            for (; i < diff.size(); ++i)
            {
                if (diff[i].first <= 0) break;
                ref_train_values.push_back(diff[i].second);
            }
            // cout<<uniq_celltypes[k1]<<"-"<<uniq_celltypes[k2]<<" "<<i<<endl;
            // cout.precision(8);
            // for (int j = 0; j < i; ++j)
            //     cout<<"score: "<<diff[j].first<<" "<<diff[j].second<<endl;
            ref_train_idxs.push_back(idx_start);
            ref_train_idxs.push_back(i);
            idx_start += i;
            // Collect common genes in top N
            for (int i = 0; i < gene_thre; ++i)
            {
                common_genes.insert(ref_width-diff[i].second-1);
                // common_genes.insert(diff[i].second);
                // cout<<diff[i].second<<endl;
                // cout<<genes[ref_width-diff[i].second-1]<<endl;
            }
            // for (auto& g : exclude_genes)
            //     if (common_genes.count(g) != 0)
            //     {
            //         cout<<g<<" "<<int(k1)<<" "<<int(k2)<<endl;
            //         exit(0);

            //     }
            // exit(0);

        }
    }
    // for (auto& g : common_genes)
    //     cout<<g<<endl;
    // exit(0);
    cout<<"common genes size: "<<common_genes.size()<<endl;
    cout<<"ref_train_idxs size: "<<ref_train_idxs.size()<<endl;
    cout<<"ref_train_values size: "<<ref_train_values.size()<<endl;
    cout<<"train time: "<<timer.toc()<<endl;

    return true;
}

DataParser::DataParser(string ref_file, string qry_file, int thread_num) : 
    ref_file(ref_file), qry_file(qry_file), thread_num(thread_num)
{
    filter_genes = false;
}

vector<char*> DataParser::getGeneIndex(string filename, string gene_index="")
{
    H5File* file = new H5File(filename.c_str(), H5F_ACC_RDONLY);
    auto group(file->openGroup("/var"));
    if (gene_index.empty())
    {
        Attribute attr(group.openAttribute("_index"));
        auto datatype  = attr.getDataType();
        attr.read(datatype, gene_index);
    }
    // cout<<"gene index: "<<gene_index<<endl;
    auto res = getDataset<char*>(group, gene_index.c_str());

    delete file;
    return res;
}

bool DataParser::findIntersectionGenes()
{
    vector<char*> ref_genes, qry_genes;
    // ref_genes = getGeneIndex(ref_file, "Symbol");
    ref_genes = getGeneIndex(ref_file);
    qry_genes = getGeneIndex(qry_file);

    genes = ref_genes;
    // cout<<genes[8277]<<" "<<genes[8278]<<" "<<genes[8279]<<endl;
    
    set<string> ref_uniq_genes(ref_genes.begin(), ref_genes.end());
    set<string> qry_uniq_genes(qry_genes.begin(), qry_genes.end());

    set<string> common_uniq_genes;
    std::set_intersection(ref_uniq_genes.begin(), ref_uniq_genes.end(), 
        qry_uniq_genes.begin(), qry_uniq_genes.end(), std::inserter(common_uniq_genes, common_uniq_genes.begin()));

    if (ref_uniq_genes.size() == qry_uniq_genes.size() && ref_uniq_genes.size() == common_uniq_genes.size())
    {
        filter_genes = false;
        cout<<"Same genes in same order, genes size: "<<common_uniq_genes.size()<<endl;
    }
    else
    {
        filter_genes = true;
        for (int i = 0; i < ref_genes.size(); ++i)
        {
            if (common_uniq_genes.count(ref_genes[i]) != 0)
                ref_gene_index.insert(i);
        }
        for (int i = 0; i < qry_genes.size(); ++i)
        {
            if (common_uniq_genes.count(qry_genes[i]) != 0)
                qry_gene_index.insert(i);
        }
        cout<<"Filter genes, qry genes reduce from "<<qry_genes.size()<<" to "<<qry_gene_index.size()
            <<", ref genes reduce from "<<ref_genes.size()<<" to "<<ref_gene_index.size()<<endl;
    }

    return true;
}

bool DataParser::loadRefData()
{
    // Load raw data from h5 file
    loadRefMatrix();

    // Logarithmize the data matrix
    std::transform(ref_data.begin(), ref_data.end(), ref_data.begin(), [](float f){ return log2(f+1);});

    // map<float, uint32> m;
    // for (auto& f : ref_data)
    //     m[f]++;
    // for (auto& [k,v] : m)
    //     cout<<"new: "<<k<<" "<<v<<endl;
    // Train data
    trainData();

    // Transform matrix format from csr to dense
    csr2dense(ref_data, ref_indptr, ref_indices, common_genes, ref_dense);
    ref_width = common_genes.size();
    for (int i = 0; i < ref_width; ++i)
        cout<<ref_dense[i]<<" ";
    cout<<endl;
    
    raw_data.ref_width  = ref_width;
    raw_data.ref_height = ref_height;

    cout<<"ref data shape: "<<ref_height <<" x "<<ref_width<<" non-zero number: "<<ref_data.size()<<endl;
    // for (int i = 0; i < 100; i++)
    //     cout<<ref_dense[i]<<" ";
    // cout<<endl;

    return true;
}

bool DataParser::csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices, int width, vector<float>& res)
{
    int height = indptr.size() - 1;
    res.resize((size_t)(width) * height, 0);

    size_t line = 0;
    for (int i = 0; i < height; ++i)
    {
        auto start = indptr[i];
        auto end = indptr[i+1];
        for (uint32 i = start; i < end; ++i)
        {
            res[line*width + indices[i]] = data[i];
        }
        line++;
    }

    return true;
}

bool DataParser::csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices, set<uint32>& cols, vector<float>& res)
{
    
    int width = cols.size();
    map<uint32, uint32> index_map;
    uint32 index = 0;
    for (auto& c : cols)
    {
        // cout<<c<<" ";
        index_map[c] = index++;
    }
    // cout<<endl;

    int height = indptr.size() - 1;
    res.resize((size_t)(width) * height, 0);

    size_t line = 0;
    for (int i = 0; i < height; ++i)
    {
        auto start = indptr[i];
        auto end = indptr[i+1];
        for (uint32 i = start; i < end; ++i)
        {
            auto raw_index = indices[i];
            if (index_map.count(raw_index) != 0)
            {
                res[line*width + index_map[raw_index]] = data[i];
            }
        }
        line++;
    }

    return true;
}

bool DataParser::loadQryData()
{
    loadQryMatrix();

    // Logarithmize the data matrix
    std::transform(qry_data.begin(), qry_data.end(), qry_data.begin(), [](float f){ return log2(f+1);});

    csr2dense(qry_data, qry_indptr, qry_indices, common_genes, qry_dense);
    qry_width = common_genes.size();
    
    raw_data.qry_height = qry_height;
    raw_data.qry_width = qry_width;

    cout<<"qry data shape: "<<qry_height <<" x "<<qry_width<<" non-zero number: "<<qry_data.size()<<endl;

    return true;
}

bool DataParser::loadQryMatrix()
{
    // Open h5 file handle
    H5File* file = new H5File(qry_file.c_str(), H5F_ACC_RDONLY);

    // Load matrix data and shape
    {
        auto group(file->openGroup("/X"));

        Attribute attr(group.openAttribute("shape"));
        auto datatype  = attr.getDataType();
        vector<uint64> shapes(2, 0);
        attr.read(datatype, shapes.data());
        qry_height = shapes[0];
        qry_width = shapes[1];
        cout<<"Qry shape: "<<qry_height<<" x "<<qry_width<<endl;

        qry_data = getDataset<float>(group, "data");
        qry_indices = getDataset<int>(group, "indices");
        qry_indptr = getDataset<int>(group, "indptr");

        set<float> m(qry_data.begin(), qry_data.end());
        cout<<"qry data uniq elements: "<<m.size()<<endl;
    }

    // clear resources
    delete file;

    return true;
}

bool DataParser::groupbyCelltypes()
{
    auto label_num = uniq_celltypes.size();

    vector< vector< uint32 > > aux_vec;
    aux_vec.resize(label_num);
    for (size_t i = 0; i < celltype_codes.size(); ++i)
    {
        aux_vec[celltype_codes[i]].push_back(i);
    }
    uint32 start = 0;
    for (auto& vec : aux_vec)
    {
        ref_ctidx.push_back(start);
        ref_ctidx.push_back(vec.size());
        start += vec.size();
        ref_ctids.insert(ref_ctids.end(), vec.begin(), vec.end());
    }
    return true;
}

bool DataParser::preprocess()
{
    Timer timer("ms");

    groupbyCelltypes();
    cout << "groupby celltypes of ref cost time(ms): " << timer.toc() << endl;

    scale(ref_dense, ref_height, ref_width, raw_data.ref);
    cout << "scale ref cost time(ms): " << timer.toc() << endl;
    vector< float >().swap(ref_dense);

    

    timer.tic();
    scale(qry_dense, qry_height, qry_width, raw_data.qry);
    cout << "scale qry cost time(ms): " << timer.toc() << endl;
    vector< float >().swap(qry_dense);

    // filter genes of datas
    timer.tic();
    // filter();
    // cout << "filter genes cost time(ms): " << timer.toc() << endl;
    // cout << "new qry size: " << raw_data.qry_height << " x " << raw_data.qry_width
    //      << endl;
    // cout << "new ref size: " << raw_data.ref_height << " x " << raw_data.ref_width
    //      << endl;

    // re-sort ref data groupby celltype
    resort();
    cout << "re-sort ref by celltype cost time(ms): " << timer.toc() << endl;

    for (int i = 0; i < ref_width/10; ++i)
        cout<<raw_data.ref[i]<<" ";
    cout<<endl;

    raw_data.ctdidx = ref_train_idxs;
    raw_data.ctdiff = ref_train_values;

    raw_data.ctids = ref_ctids;
    raw_data.ctidx = ref_ctidx;

    raw_data.ct_num = uniq_celltypes.size();

    raw_data.celltypes = uniq_celltypes;

    return true;
}

bool DataParser::scale(vector< float >& src, const uint32 rows, const uint32 cols,
                     vector< uint16 >& dest)
{
    dest.resize(src.size(), 0);

    std::mutex m;
    auto       task = [&](size_t start, size_t rows, size_t cols)
    {
        size_t max_index = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            set< float > uniq;
            for (size_t j = 0; j < cols; ++j)
                uniq.insert(src[start + i * cols + j]);
            assert(uniq.size() < 65536);

            vector< float >                order(uniq.begin(), uniq.end());
            unordered_map< float, uint16 > index;
            for (uint16 j = 0; j < order.size(); ++j)
            {
                index[order[j]] = j;
            }
            for (size_t j = 0; j < cols; ++j)
                dest[start + i * cols + j] = index[src[start + i * cols + j]];
            max_index = max(max_index, index.size());
        }

        // std::lock_guard<std::mutex> lg(m);
        // cout<<"max index: "<<max_index<<endl;
    };
    vector< thread > threads;
    size_t           start = 0;
    size_t           step  = rows / thread_num;
    if (rows % thread_num != 0)
        step++;
    for (int i = 0; i < thread_num; ++i)
    {
        start = i * step * cols;
        if (i == (thread_num - 1))
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

bool DataParser::filterGenes(vector< uint16 >& src, const uint32 rows, const uint32 cols,
                            set< uint32 >& genes)
{
    vector< uint16 > dest;
    dest.resize(size_t(rows) * genes.size(), 0);

    auto task = [&](size_t start, size_t rows, size_t cols)
    {
        for (size_t i = 0; i < rows; ++i)
        {
            uint32 nj = 0;
            for (size_t j = 0; j < cols; ++j)
            {
                if (genes.count(cols - j - 1) == 0)
                    continue;
                dest[(start + i) * genes.size() + nj] = src[(start + i) * cols + j];
                nj++;
            }
        }
    };
    vector< thread > threads;
    size_t           start = 0;
    size_t           step  = rows / thread_num;
    if (rows % thread_num != 0)
        step++;
    for (int i = 0; i < thread_num; ++i)
    {
        start = i * step;
        if (i == (thread_num - 1))
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

void DataParser::filter()
{
    set< uint32 > uniq_genes;
    int           gene_thre = round(500 * pow((2 / 3.0), log2(2)));
    for (int i = 0; i < label_num; ++i)
    {
        for (int j = 0; j < label_num; ++j)
        {
            if (i == j)
                continue;
            int pos = ref_train_idxs[(i * label_num + j) * 2];
            int len = ref_train_idxs[(i * label_num + j) * 2 + 1];
            if (len > gene_thre)
                len = gene_thre;
            uniq_genes.insert(ref_train_values.begin() + pos, ref_train_values.begin() + pos + len);
            // cout<<"temp uniq genes size: "<<uniq_genes.size()<<endl;
        }
    }
    cout << "useful genes: " << uniq_genes.size() << endl;
    unordered_map< uint32, uint32 > gene_map;
    size_t                          idx = 0;
    for (auto v : uniq_genes)
        gene_map[v] = idx++;

    // re-construct trained data by filtering genes
    vector< uint32 > _ctdiff;
    vector< uint32 > _ctdidx;
    size_t           start = 0;
    for (size_t i = 0; i < ref_train_idxs.size(); i += 2)
    {
        auto s = ref_train_idxs[i];
        auto l = ref_train_idxs[i + 1];
        if (s == 0 && l == 0)
        {
            _ctdidx.push_back(0);
            _ctdidx.push_back(0);
        }
        else
        {
            size_t ns = _ctdiff.size(), nl = 0;
            for (size_t j = s; j < s + l; j++)
            {
                if (uniq_genes.count(ref_train_values[j]) == 0)
                    continue;
                _ctdiff.push_back(gene_map[ref_train_values[j]]);
                nl++;
            }
            _ctdidx.push_back(ns);
            _ctdidx.push_back(nl);
        }
    }
    _ctdiff.swap(ref_train_values);
    _ctdidx.swap(ref_train_idxs);

    
    // filter genes for ref data
    filterGenes(raw_data.ref, ref_height, ref_width, uniq_genes);

    // filter genes for qry data
    filterGenes(raw_data.qry, qry_height, qry_width, uniq_genes);

    raw_data.ref_width  = uniq_genes.size();
    raw_data.qry_width = uniq_genes.size();
    raw_data.ref_height = ref_height;
    raw_data.qry_height = qry_height;
}

void DataParser::resort()
{
    vector< uint16 > dest;
    dest.resize(raw_data.ref.size(), 0);
    cout<<dest.size()<<endl;
    // for (int i = 0; i < label_num; ++i)
    // {
    //     cout<<i<<" "<<ref_ctidx[i*2]<<" "<<ref_ctidx[i*2+1]<<endl;
    // }
    // cout<<ref_ctids.size()<<endl;

    auto width = raw_data.ref_width;
    auto task = [&](size_t start)
    {
        for (size_t i = start; i < label_num; i += thread_num)
        {
            size_t pos = ref_ctidx[i * 2];
            size_t len = ref_ctidx[i * 2 + 1];
            size_t idx = pos * width;
            for (size_t j = pos; j < pos + len; ++j)
            {
                for (size_t k = 0; k < width; ++k)
                    dest[idx++] = raw_data.ref[ref_ctids[j] * width + k];
            }
        }
    };
    vector< thread > threads;
    for (int i = 0; i < thread_num; ++i)
    {
        thread th(task, i);
        threads.push_back(std::move(th));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    dest.swap(raw_data.ref);
    for (uint32 i = 0; i < ref_ctids.size(); ++i)
        ref_ctids[i] = i;
}