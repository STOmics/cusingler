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
using namespace std;

#include "H5Cpp.h"
using namespace H5;

bool readObs(H5File* file, InputData& data)
{
    // StrType str_type(PredType::C_S1, H5T_VARIABLE);
    // str_type.setCset(H5T_CSET_UTF8);

    DataSet   dataset   = DataSet(file->openDataSet("/obs_key"));
    auto      datatype  = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    hsize_t   dims[1];
    dataspace.getSimpleExtentDims(dims, NULL);
    char* obs_key[dims[0]];
    dataset.read(obs_key, datatype);

    dataset = DataSet(file->openDataSet("/obs_value"));
    char* obs_value[dims[0]];
    dataset.read(&obs_value[0], datatype);

    for (size_t i = 0; i < dims[0]; ++i)
    {
        // data.obs.insert({obs_key[i], obs_value[i]});
        data.obs_keys.push_back(obs_key[i]);
        data.obs_values.push_back(obs_value[i]);
    }

    cout << "obs size: " << data.obs_keys.size() << endl;
    // for (auto&[k, v] : data.obs)
    //     cout<<k<<" "<<v<<endl;

    return true;
}

bool readRef(H5File* file, InputData& data)
{
    auto& ref = data.ref;

    DataSet   dataset   = DataSet(file->openDataSet("/ref"));
    auto      datatype  = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    int       rank      = dataspace.getSimpleExtentNdims();
    hsize_t   dims[rank];
    dataspace.getSimpleExtentDims(dims, NULL);

    size_t ref_size = 1;
    for (int i = 0; i < rank; ++i)
        ref_size *= dims[i];
    ref.resize(ref_size);
    dataset.read(&ref[0], datatype);

    data.ref_cell_num = dims[0];
    data.ref_gene_num = dims[1];
    cout << "ref size: " << ref.size() << " " << dims[0] << " x " << dims[1] << endl;

    // map<float, uint32> m;
    // for (auto& f : ref)
    //     m[f]++;
    // for (auto& [k,v] : m)
    //     cout<<"old: "<<k<<" "<<v<<endl;

    return true;
}

bool readTest(H5File* file, InputData& data)
{
    auto& test = data.test;

    DataSet   dataset   = DataSet(file->openDataSet("/test"));
    auto      datatype  = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    int       rank      = dataspace.getSimpleExtentNdims();
    hsize_t   dims[rank];
    dataspace.getSimpleExtentDims(dims, NULL);

    size_t test_size = 1;
    for (int i = 0; i < rank; ++i)
        test_size *= dims[i];
    test.resize(test_size);
    dataset.read(&test[0], datatype);

    data.test_cell_num = dims[0];
    data.test_gene_num = dims[1];
    cout << "test size: " << test.size() << " " << dims[0] << " x " << dims[1] << endl;

    return true;
}

bool readLabels(H5File* file, InputData& data)
{
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

    Attribute attr(dataset.openAttribute("order"));
    datatype  = attr.getDataType();
    dataspace = attr.getSpace();
    dataspace.getSimpleExtentDims(dims, NULL);
    char* orders[dims[0]];
    attr.read(datatype, orders);

    for (int i = 0; i < dims[0]; ++i)
    {
        data.celltypes.push_back(orders[i]);
    }
    cout << "celltypes size: " << data.celltypes.size() << endl;

    return true;
}

bool readTrained(H5File* file, InputData& data)
{
    // aux function for traveling datasets of '/trained'
    auto file_info =
        [](hid_t loc_id, const char* name, const H5L_info_t* linfo, void* opdata)
    {
        auto dataset = H5Dopen2(loc_id, name, H5P_DEFAULT);
        auto ids     = reinterpret_cast< vector< string >* >(opdata);
        ids->push_back(name);
        H5Dclose(dataset);
        return 0;
    };
    vector< string > ids;
    Group            group = Group(file->openGroup("/trained"));
    H5Literate(group.getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, &ids);

    auto& trained = data.trained;
    for (auto& id : ids)
    {
        vector< double > aux_list;

        DataSet   dataset   = DataSet(group.openDataSet(id));
        auto      datatype  = dataset.getDataType();
        DataSpace dataspace = dataset.getSpace();
        int       rank      = dataspace.getSimpleExtentNdims();
        hsize_t   dims[rank];
        dataspace.getSimpleExtentDims(dims, NULL);

        size_t test_size = 1;
        for (int i = 0; i < rank; ++i)
            test_size *= dims[i];
        aux_list.resize(test_size);
        dataset.read(&aux_list[0], datatype);

        trained.insert({ id, aux_list });
    }

    cout << "trained data size: " << trained.size() << endl;

    return true;
}

bool readInput(string& filename, InputData& data)
{
    // open h5 file handle
    H5File* file = new H5File(filename.c_str(), H5F_ACC_RDONLY);

    // read obs key and value, then construct map
    readObs(file, data);

    // read ref from matrix to list
    readRef(file, data);

    // read query data from matrix to list
    readTest(file, data);

    // read labels
    readLabels(file, data);

    // read trained data
    readTrained(file, data);

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

    for (auto& [k1, v1] : median_map)
    {
        for (auto& [k2, v2] : median_map)
        {
            if (k1 == k2)
            {
                // padding zero
                ref_idxs.push_back(0);
                ref_idxs.push_back(0);
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
                ref_values.push_back(diff[i].second);
            }
            // cout<<uniq_celltypes[k1]<<"-"<<uniq_celltypes[k2]<<" "<<i<<endl;
            // for (int j = 0; j < i; ++j)
            //     cout<<"score: "<<diff[j].first<<" "<<diff[j].second<<endl;
            ref_idxs.push_back(idx_start);
            ref_idxs.push_back(i);
            idx_start += i;
            // Collect common genes in top N
            for (int i = 0; i < gene_thre; ++i)
            {
                common_genes.insert(diff[i].second);
            }

        }
    }
    cout<<"common genes size: "<<common_genes.size()<<endl;
    cout<<"ref_idxs size: "<<ref_idxs.size()<<endl;
    cout<<"ref_values size: "<<ref_values.size()<<endl;
    cout<<"train time: "<<timer.toc()<<endl;

    return true;
}

DataParser::DataParser(string ref_file, string qry_file) : 
    ref_file(ref_file), qry_file(qry_file)
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
    csr2dense(ref_data, ref_indptr, ref_indices, ref_width, ref_dense);

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

bool DataParser::loadQryData()
{
    loadQryMatrix();

    // Logarithmize the data matrix
    std::transform(qry_data.begin(), qry_data.end(), qry_data.begin(), [](float f){ return log2(f+1);});

    csr2dense(qry_data, qry_indptr, qry_indices, qry_width, qry_dense);

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