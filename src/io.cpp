/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "io.h"
#include "timer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>
using namespace std;

// #include "H5Cpp.h"
// using namespace H5;

// template <typename T> vector<T> getDataset(Group& group, string name)
// {
//     auto dataset   = DataSet(group.openDataSet(name.c_str()));
//     auto datatype  = dataset.getDataType();
//     auto dataspace = dataset.getSpace();
//     int  rank      = dataspace.getSimpleExtentNdims();
//     // hsize_t dims[rank];
//     vector<hsize_t> dims(rank);
//     dataspace.getSimpleExtentDims(&dims[0], NULL);

//     uint64    size = dims[0];
//     vector<T> data;
//     data.resize(size);
//     dataset.read(data.data(), datatype);

//     return data;
// }

bool DataParser::loadRefMatrix(string ref_file)
{
//     // Open h5 file handle
//     H5File* file = new H5File(ref_file.c_str(), H5F_ACC_RDONLY);

//     // Load matrix data and shape
//     {
//         auto group(file->openGroup("/X"));

//         Attribute attr(group.openAttribute("shape"));
//         auto      datatype = attr.getIntType();
//         size_t    byteSize = datatype.getSize();
//         if (byteSize == 1)
//         {
//             vector<uint8> shapes(2, 0);
//             attr.read(datatype, shapes.data());
//             ref_height = shapes[0];
//             ref_width  = shapes[1];
//         }
//         else if (byteSize == 2)
//         {
//             vector<uint16> shapes(2, 0);
//             attr.read(datatype, shapes.data());
//             ref_height = shapes[0];
//             ref_width  = shapes[1];
//         }
//         else if (byteSize == 4)
//         {
//             vector<uint32> shapes(2, 0);
//             attr.read(datatype, shapes.data());
//             ref_height = shapes[0];
//             ref_width  = shapes[1];
//         }
//         else if (byteSize == 8)
//         {
//             vector<uint64> shapes(2, 0);
//             attr.read(datatype, shapes.data());
//             ref_height = shapes[0];
//             ref_width  = shapes[1];
//         }

//         cout << "Ref shape: " << ref_height << " x " << ref_width << endl;

//         ref_data    = getDataset<float>(group, "data");
//         ref_indices = getDataset<int>(group, "indices");
//         ref_indptr  = getDataset<int>(group, "indptr");
//     }

//     // Load celltypes of per cell
//     {
//         bool  is_dataset = false;
//         Group group;
//         if (file->nameExists("/obs/celltype"))
//         {
//             group = file->openGroup("/obs/celltype");
//         }
//         else if (file->nameExists("/obs/ClusterName"))
//         {
//             group = file->openGroup("/obs/ClusterName");
//         }
//         else if (file->nameExists("/obsm/annotation_au/celltype"))
//         {
//             is_dataset = true;

//             group                   = file->openGroup("/obsm/annotation_au");
//             auto               temp = getDataset<char*>(group, "celltype");
//             map<string, uint8> m;
//             for (auto s : temp)
//             {
//                 if (m.count(s) == 0)
//                 {
//                     m[s] = m.size();
//                     char* t = strdup(s);
//                     uniq_celltypes.push_back(t);
//                 }
//                 celltype_codes.push_back(m[s]);
//             }
//             label_num = uniq_celltypes.size();
//         }
//         else
//         {
//             cerr << "Fail to load celltypes of reference file." << endl;
//             exit(-1);
//         }

//         if (!is_dataset)
//         {
//             celltype_codes = getDataset<uint8>(group, "codes");
//             auto temp = getDataset<char*>(group, "categories");
//             std::copy(temp.begin(), temp.end(), std::back_inserter(uniq_celltypes));
//             label_num      = uniq_celltypes.size();
//         }
//     }

//     // clear resources
//     delete file;

    return true;
}

bool DataParser::trainData()
{
    auto task = [](vector<vector<float>>& matrix, const int start, const int end,
                   const int len, vector<float>& res)
    {
        for (int i = start; i < end; ++i)
        {
            auto& cols = matrix[i];
            cols.resize(len, 0);
            float median = 0;
            if (cols.size() % 2 == 0)
            {
                std::sort(cols.begin(), cols.end());
                median = (cols[cols.size() / 2] + cols[cols.size() / 2 - 1]) / 2;
            }
            else
            {
                std::nth_element(cols.begin(), cols.begin() + cols.size() / 2,
                                 cols.end());
                median = cols[cols.size() / 2];
            }
            res[i] = median;
        }
    };

    // Calculate median gene value for each celltype
    uint32 step = ref_width / thread_num;
    if ((ref_width % thread_num) != 0)
        step++;
    map<uint8, vector<float>> median_map;
    for (uint32 i = 0; i < ref_ctidx.size(); i += 2)
    {
        int  ct  = i / 2;
        auto pos = ref_ctidx[i];
        auto len = ref_ctidx[i + 1];

        // Collect sub 2d-array
        vector<vector<float>> sub_ref(ref_width);
        for (uint32 idx = pos; idx < pos + len; ++idx)
        {
            auto start = ref_indptr[idx];
            auto end   = ref_indptr[idx + 1];
            for (int i = start; i < end; ++i)
            {
                sub_ref[ref_indices[i]].push_back(ref_data[i]);
            }
        }

        // Calculating median value using multi-threading
        vector<float>  median(ref_width, 0);
        vector<thread> threads;
        for (int i = 0; i < thread_num; ++i)
        {
            int    start = i * step;
            int    end   = min((i + 1) * step, ref_width);
            thread th(task, std::ref(sub_ref), start, end, len, std::ref(median));
            threads.push_back(std::move(th));
        }
        for (auto& th : threads)
        {
            th.join();
        }

        median_map.insert({ ct, median });
    }

    // Calculate difference for each two celltypes
    uint32 idx_start     = 0;
    size_t common_gene_n = round(500 * pow((2 / 3.0), log2(uniq_celltypes.size())));
    size_t thre_gene_n   = round(500 * pow((2 / 3.0), log2(2)));

    vector<int> my_scores;
    for (auto& tmp1 : median_map)
    {
        auto& k1 = tmp1.first;
        auto& v1 = tmp1.second;
        for (auto& tmp2 : median_map)
        {
            auto& k2 = tmp2.first;
            auto& v2 = tmp2.second;
            if (k1 == k2)
            {
                // padding zero
                ref_train_idxs.push_back(0);
                ref_train_idxs.push_back(0);
                continue;
            }
            // Get diff of two array
            vector<pair<int, uint32>> diff;
            for (uint32 i = 0; i < ref_width; ++i)
            {
                diff.push_back({ int(round((v1[i] - v2[i]) * 1e6)), ref_width - i - 1 });
            }

            // Sort by ascending order
            std::sort(diff.begin(), diff.end(), std::greater<pair<int, uint32>>());

            // Only need the score > 0
            uint32 i = 0;
            for (; i < diff.size(); ++i)
            {
                if (diff[i].first <= 0)
                    break;
                ref_train_values.push_back(ref_width - diff[i].second - 1);
                my_scores.push_back(diff[i].first);
                if (i < thre_gene_n)
                    thre_genes.insert(ref_width - diff[i].second - 1);
            }

            ref_train_idxs.push_back(idx_start);
            ref_train_idxs.push_back(i);
            idx_start += i;
            // Collect common genes in top N
            for (uint32 i = 0; i < common_gene_n; ++i)
            {
                common_genes.insert(ref_width - diff[i].second - 1);
            }
        }
    }

    return true;
}

DataParser::DataParser(int thread_num)
    : thread_num(thread_num)
{
    filter_genes = false;
}

bool DataParser::prepareData(string ref_file, string qry_file)
{
    // ref_genes = getGeneIndex(ref_file, "");
    // qry_genes = getGeneIndex(qry_file, "");

    // // Load raw data from h5 file
    // loadRefMatrix(ref_file);
    // loadQryMatrix(qry_file);

    return true;
}


vector<string> DataParser::getGeneIndex(string filename, string gene_index = "")
{
    // H5File* file = new H5File(filename.c_str(), H5F_ACC_RDONLY);
    // auto    group(file->openGroup("/var"));
    // if (gene_index.empty())
    // {
    //     Attribute attr(group.openAttribute("_index"));
    //     auto      datatype = attr.getDataType();
    //     attr.read(datatype, gene_index);
    //     // FIXME: not robust
    //     if (gene_index == "Gene_ID")
    //         gene_index = "Symbol";
    // }
    // auto temp = getDataset<char*>(group, gene_index.c_str());
    // vector<string> res;
    // std::copy(temp.begin(), temp.end(), std::back_inserter(res));


    // delete file;
    // return res;
    return {};
}

bool DataParser::findIntersectionGenes()
{
    set<string> ref_uniq_genes(ref_genes.begin(), ref_genes.end());
    set<string> qry_uniq_genes(qry_genes.begin(), qry_genes.end());

    set<string> common_uniq_genes;
    std::set_intersection(ref_uniq_genes.begin(), ref_uniq_genes.end(),
                          qry_uniq_genes.begin(), qry_uniq_genes.end(),
                          std::inserter(common_uniq_genes, common_uniq_genes.begin()));

    if (ref_uniq_genes.size() == qry_uniq_genes.size()
        && ref_uniq_genes.size() == common_uniq_genes.size())
    {
        filter_genes = false;
        cout << "Same genes in same order, genes size: " << common_uniq_genes.size()
             << endl;
    }
    else
    {
        if (common_uniq_genes.empty())
        {
            cerr << "No intersection genes in two file, please check input data!" << endl;
            exit(-1);
        }
        filter_genes  = true;
        int new_index = 0;
        for (uint32 i = 0; i < ref_genes.size(); ++i)
        {
            if (common_uniq_genes.count(ref_genes[i]) != 0)
                ref_gene_index.insert({ i, new_index++ });
        }
        new_index = 0;
        for (uint32 i = 0; i < qry_genes.size(); ++i)
        {
            if (common_uniq_genes.count(qry_genes[i]) != 0)
                qry_gene_index.insert({ i, new_index++ });
        }
        cout << "Filter genes, qry genes reduce from " << qry_genes.size() << " to "
             << qry_gene_index.size() << ", ref genes reduce from " << ref_genes.size()
             << " to " << ref_gene_index.size() << endl;
    }

    return true;
}

bool DataParser::loadRefData()
{
    
    // Logarithmize the data matrix
    // std::transform(ref_data.begin(), ref_data.end(), ref_data.begin(), [](float f){
    // return log2(f+1);});

    // Groupby cell index through celltypes and resort ref csr data
    groupbyCelltypes();
    resort();

    if (filter_genes)
    {
        // removeCols(data, indptr, indices, gene_index);
        removeCols(ref_data, ref_indptr, ref_indices, ref_gene_index);
        ref_gene_index.clear();
    }

    return true;
}

bool DataParser::csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                           int width, vector<float>& res)
{
    int height = indptr.size() - 1;
    res.resize(( size_t )( width )*height, 0);

    size_t line = 0;
    for (int i = 0; i < height; ++i)
    {
        auto start = indptr[i];
        auto end   = indptr[i + 1];
        for (int i = start; i < end; ++i)
        {
            res[line * width + indices[i]] = data[i];
        }
        line++;
    }

    return true;
}

bool DataParser::csr2dense(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                           set<uint32>& cols, vector<uint16>& res, uint64& max_uniq_gene)
{

    int                 width = cols.size();
    map<uint32, uint32> index_map;
    uint32              index = 0;
    for (auto& c : cols)
    {
        index_map[c] = index++;
    }

    int height = indptr.size() - 1;
    res.clear();
    res.resize(( size_t )( width )*height, 0);
    std::mutex stat_mutex;
    auto       task = [&](int line_start, int line_end)
    {
        uint64 curr_uniq_gene = 0;
        for (int line = line_start; line < line_end; ++line)
        {
            auto start = indptr[line];
            auto end   = indptr[line + 1];

            set<float> uniq;
            for (int i = start; i < end; ++i)
            {
                auto raw_index = indices[i];
                if (index_map.count(raw_index) != 0)
                {
                    uniq.insert(data[i]);
                }
            }

            curr_uniq_gene = std::max<uint64>(uniq.size() + 1, curr_uniq_gene);

            vector<float>                order(uniq.begin(), uniq.end());
            unordered_map<float, uint16> index;
            for (uint16 j = 0; j < order.size(); ++j)
            {
                // There is not exists 0 in csr format, so start from 1
                index[order[j]] = j + 1;
            }

            for (int i = start; i < end; ++i)
            {
                auto raw_index = indices[i];
                if (index_map.count(raw_index) != 0)
                {
                    res[line * width + index_map[raw_index]] = index[data[i]];
                }
            }
        }

        std::lock_guard<std::mutex> lg(stat_mutex);
        max_uniq_gene = max(curr_uniq_gene, max_uniq_gene);
    };

    vector<thread> threads;
    int            step = height / thread_num;
    if ((height % thread_num) != 0)
        step++;
    for (int i = 0; i < thread_num; ++i)
    {
        int    start = i * step;
        int    end   = min((i + 1) * step, height);
        thread th(task, start, end);
        threads.push_back(std::move(th));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    return true;
}

bool DataParser::csrFilter(vector<float>& data, vector<int>& indptr, vector<int>& indices,
                           set<uint32>& cols, vector<uint16>& res_data,
                           vector<int>& res_indptr, vector<int>& res_indices,
                           uint64& max_uniq_gene)
{
    map<uint32, uint32> index_map;
    uint32              index = 0;
    for (auto& c : cols)
    {
        index_map[c] = index++;
    }

    res_data.clear();
    res_indptr.clear();
    res_indptr.push_back(0);
    res_indices.clear();

    uint64 curr_uniq_gene = 0;
    for (uint32 j = 0; j < indptr.size() - 1; j++)
    {
        auto start = indptr[j];
        auto end   = indptr[j + 1];

        set<float> uniq;
        for (int i = start; i < end; ++i)
        {
            auto raw_index = indices[i];
            if (index_map.count(raw_index) != 0)
            {
                uniq.insert(data[i]);
            }
        }
        curr_uniq_gene = std::max<uint64>(uniq.size() + 1, curr_uniq_gene);
        vector<float>                order(uniq.begin(), uniq.end());
        unordered_map<float, uint16> index;
        for (uint16 i = 0; i < order.size(); ++i)
        {
            // There is not exists 0 in csr format, so start from 1
            index[order[i]] = i + 1;
        }

        for (int i = start; i < end; ++i)
        {
            if (index_map.count(indices[i]) == 0)
                continue;
            res_indices.push_back(index_map[indices[i]]);
            res_data.push_back(index[data[i]]);
        }
        res_indptr.push_back(res_data.size());
    }
    max_uniq_gene = max(max_uniq_gene, curr_uniq_gene);

    return true;
}

bool DataParser::loadQryData()
{
    
    // Logarithmize the data matrix
    // std::transform(qry_data.begin(), qry_data.end(), qry_data.begin(), [](float f){
    // return log2(f+1);});

    if (filter_genes)
    {
        removeCols(qry_data, qry_indptr, qry_indices, qry_gene_index);
        qry_gene_index.clear();
    }
    return true;
}

bool DataParser::loadQryMatrix(string qry_file)
{
    // // Open h5 file handle
    // H5File* file = new H5File(qry_file.c_str(), H5F_ACC_RDONLY);

    // // Load matrix data and shape
    // {
    //     auto group(file->openGroup("/X"));

    //     Attribute attr(group.openAttribute("shape"));
    //     auto      datatype = attr.getIntType();
    //     size_t    byteSize = datatype.getSize();
    //     if (byteSize == 1)
    //     {
    //         vector<uint8> shapes(2, 0);
    //         attr.read(datatype, shapes.data());
    //         qry_height = shapes[0];
    //         qry_width  = shapes[1];
    //     }
    //     else if (byteSize == 2)
    //     {
    //         vector<uint16> shapes(2, 0);
    //         attr.read(datatype, shapes.data());
    //         qry_height = shapes[0];
    //         qry_width  = shapes[1];
    //     }
    //     else if (byteSize == 4)
    //     {
    //         vector<uint32> shapes(2, 0);
    //         attr.read(datatype, shapes.data());
    //         qry_height = shapes[0];
    //         qry_width  = shapes[1];
    //     }
    //     else if (byteSize == 8)
    //     {
    //         vector<uint64> shapes(2, 0);
    //         attr.read(datatype, shapes.data());
    //         qry_height = shapes[0];
    //         qry_width  = shapes[1];
    //     }

    //     cout << "Qry shape: " << qry_height << " x " << qry_width << endl;

    //     DataSet dataset(group.openDataSet("data"));
    //     auto t = dataset.getFloatType();
    //     byteSize = t.getSize();
    //     if (byteSize == 4)
    //     {
    //         qry_data    = getDataset<float>(group, "data");
    //     }
    //     else if (byteSize == 8)
    //     {
    //         auto temp = getDataset<double>(group, "data");
    //         std::copy(temp.begin(), temp.end(), std::back_inserter(qry_data));
    //     }

    //     qry_indices = getDataset<int>(group, "indices");
    //     qry_indptr  = getDataset<int>(group, "indptr");

    //     // set<float> m(qry_data.begin(), qry_data.end());
    //     // cout<<"qry data uniq elements: "<<m.size()<<endl;
    // }

    // // Load per cell names
    // {
    //     auto group(file->openGroup("/obs"));
    //     auto temp = getDataset<char*>(group, "_index");
    //     std::copy(temp.begin(), temp.end(), std::back_inserter(qry_cellnames));
    // }

    // // clear resources
    // delete file;

    return true;
}

bool DataParser::groupbyCelltypes()
{
    auto label_num = uniq_celltypes.size();

    vector<vector<uint32>> aux_vec;
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

    raw_data.ctdidx = ref_train_idxs;
    raw_data.ctdiff = ref_train_values;

    raw_data.ctidx = ref_ctidx;

    raw_data.ct_num = uniq_celltypes.size();

    for (auto s : uniq_celltypes)
        raw_data.celltypes.push_back(string(s));

    raw_data.qry_cellnames = qry_cellnames;

    return true;
}

// Resort csr lines of ref, groupby celltypes
void DataParser::resort()
{
    vector<int>   _indices;
    vector<int>   _indptr{ 0 };
    vector<float> _data;

    for (uint32 i = 0; i < ref_ctidx.size(); i += 2)
    {
        auto pos = ref_ctidx[i];
        auto len = ref_ctidx[i + 1];
        for (uint32 j = pos; j < pos + len; j++)
        {
            auto line_num = ref_ctids[j];
            auto start    = ref_indptr[line_num];
            auto end      = ref_indptr[line_num + 1];
            _indptr.push_back(_indptr.back() + end - start);
            _indices.insert(_indices.end(), ref_indices.begin() + start,
                            ref_indices.begin() + end);
            _data.insert(_data.end(), ref_data.begin() + start, ref_data.begin() + end);
        }
    }

    _indices.swap(ref_indices);
    _indptr.swap(ref_indptr);
    _data.swap(ref_data);

    ref_ctids.clear();
}

// Resort csr lines of ref, groupby celltypes
void DataParser::removeCols(vector<float>& data, vector<int>& indptr,
                            vector<int>& indices, map<uint32, uint32>& colsMap)
{
    vector<int>   _indices;
    vector<int>   _indptr{ 0 };
    vector<float> _data;

    for (uint32 j = 0; j < indptr.size() - 1; j++)
    {
        auto start = indptr[j];
        auto end   = indptr[j + 1];
        for (int i = start; i < end; ++i)
        {
            if (colsMap.count(indices[i]) == 0)
                continue;
            _indices.push_back(colsMap[indices[i]]);
            _data.push_back(data[i]);
        }
        _indptr.push_back(_data.size());
    }

    _indices.swap(indices);
    _indptr.swap(indptr);
    _data.swap(data);
}

// Transform matrix format from csr to dense
bool DataParser::generateDenseMatrix(int step, uint64& max_uniq_gene)
{
    set<uint32> gene_set;
    if (step == 0)
    {
        // Use common genes for step score()
        gene_set = common_genes;
    }
    else if (step == 1)
    {
        // Use threshold genes for step finetune()
        gene_set = thre_genes;

        map<uint32, uint32> index_map;
        uint32              index = 0;
        for (auto& c : gene_set)
        {
            index_map[c] = index++;
        }

        raw_data.ctdidx.clear();
        raw_data.ctdiff.clear();
        size_t start = 0;
        for (uint32 i = 0; i < ref_train_idxs.size(); i += 2)
        {
            auto pos = ref_train_idxs[i];
            auto len = ref_train_idxs[i + 1];
            if (pos == 0 && len == 0)
            {
                raw_data.ctdidx.push_back(0);
                raw_data.ctdidx.push_back(0);
                continue;
            }
            for (uint32 j = pos; j < pos + len; ++j)
            {
                if (gene_set.count(ref_train_values[j]) == 0)
                    continue;
                raw_data.ctdiff.push_back(index_map[ref_train_values[j]]);
            }
            raw_data.ctdidx.push_back(start);
            raw_data.ctdidx.push_back(raw_data.ctdiff.size() - start);
            start = raw_data.ctdiff.size();
        }
    }

    // csr2dense(ref_data, ref_indptr, ref_indices, gene_set, raw_data.ref,
    // max_uniq_gene);
    csrFilter(ref_data, ref_indptr, ref_indices, gene_set, raw_data.ref_data,
              raw_data.ref_indptr, raw_data.ref_indices, max_uniq_gene);

    ref_width           = gene_set.size();
    raw_data.ref_width  = ref_width;
    raw_data.ref_height = ref_height;

    cout << "ref data shape: " << ref_height << " x " << ref_width
         << " non-zero number: " << ref_data.size() << endl;

    // csr2dense(qry_data, qry_indptr, qry_indices, gene_set, raw_data.qry,
    // max_uniq_gene);
    csrFilter(qry_data, qry_indptr, qry_indices, gene_set, raw_data.qry_data,
              raw_data.qry_indptr, raw_data.qry_indices, max_uniq_gene);

    qry_width           = gene_set.size();
    raw_data.qry_height = qry_height;
    raw_data.qry_width  = qry_width;

    cout << "qry data shape: " << qry_height << " x " << qry_width
         << " non-zero number: " << qry_data.size() << endl;

    return true;
}

PyDataParser::PyDataParser(int thread_num) : 
    DataParser(thread_num)
{

}

bool PyDataParser::prepareData(uint32 ref_height_, uint32 ref_width_,
        vector<float>& ref_data_, vector<int>& ref_indices_, vector<int>& ref_indptr_,
        uint32 qry_height_, uint32 qry_width_,
        vector<float>& qry_data_, vector<int>& qry_indices_, vector<int>& qry_indptr_,
        vector<string>& codes_, vector<int>& celltypes_,
        vector<string>& cellnames_, vector<string>& ref_geneidx_, vector<string>& qry_geneidx_)
{
    ref_height = ref_height_;
    ref_width = ref_width_;
    ref_data = ref_data_;
    ref_indices = ref_indices_;
    ref_indptr = ref_indptr_;

    qry_height = qry_height_;
    qry_width = qry_width_;
    qry_data = qry_data_;
    qry_indices = qry_indices_;
    qry_indptr = qry_indptr_;

    uniq_celltypes = codes_;
    // celltype_codes = celltypes_;
    std::copy(celltypes_.begin(), celltypes_.end(), std::back_inserter(celltype_codes));
    label_num      = uniq_celltypes.size();

    qry_cellnames = cellnames_;
    ref_genes = ref_geneidx_;
    qry_genes = qry_geneidx_;

    return true;
}