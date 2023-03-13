/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#include "io.h"

#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "H5Cpp.h"
using namespace H5;

bool readObs(H5File* file, InputData& data)
{
    vector<string> obs_key, obs_value;

    DataSet dataset = DataSet(file->openDataSet("/obs_key"));
    auto   datatype   = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1];
    dataspace.getSimpleExtentDims(dims, NULL);
    obs_key.resize(dims[0]);
    dataset.read(&obs_key[0], datatype);

    dataset = DataSet(file->openDataSet("/obs_value"));
    obs_value.resize(dims[0]);
    dataset.read(&obs_value[0], datatype);

    size_t pos1, pos2;
    for (size_t i = 0; i < obs_key.size(); ++i)
    {
        pos1 = obs_key[i].find('\0');
        pos2 = obs_value[i].find('\0');
        data.obs.insert({obs_key[i].substr(0,pos1), obs_value[i].substr(0,pos2)});
    }

    cout<<"obs size: "<<data.obs.size()<<endl;

    return true;
}

bool readRef(H5File* file, InputData& data)
{
    auto& ref = data.ref;

    DataSet dataset = DataSet(file->openDataSet("/ref"));
    auto   datatype   = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    hsize_t dims[rank];
    dataspace.getSimpleExtentDims(dims, NULL);

    size_t ref_size = 1;
    for (int i = 0; i < rank; ++i)
        ref_size *= dims[i];
    ref.resize(ref_size);
    dataset.read(&ref[0], datatype);

    cout<<"ref size: "<<ref.size()<<endl;

    return true;
}

bool readTest(H5File* file, InputData& data)
{
    auto& test = data.test;

    DataSet dataset = DataSet(file->openDataSet("/test"));
    auto   datatype   = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    hsize_t dims[rank];
    dataspace.getSimpleExtentDims(dims, NULL);

    size_t test_size = 1;
    for (int i = 0; i < rank; ++i)
        test_size *= dims[i];
    test.resize(test_size);
    dataset.read(&test[0], datatype);

    cout<<"test size: "<<test.size()<<endl;

    return true;
}

bool readCelltype(H5File* file, InputData& data)
{
    auto& celltype = data.celltype;

    DataSet dataset = DataSet(file->openDataSet("/tmp"));
    auto   datatype   = dataset.getDataType();
    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    hsize_t dims[rank];
    dataspace.getSimpleExtentDims(dims, NULL);

    size_t test_size = 1;
    for (int i = 0; i < rank; ++i)
        test_size *= dims[i];
    celltype.resize(test_size);
    dataset.read(&celltype[0], datatype);

    cout<<"celltype size: "<<celltype.size()<<endl;

    return true;
}



bool readTrained(H5File* file, InputData& data)
{
    // aux function for traveling datasets of '/trained'
    auto file_info = [](hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata)
    {
        auto dataset = H5Dopen2(loc_id, name, H5P_DEFAULT);
        auto ids = reinterpret_cast< vector<string>* >(opdata);
        ids->push_back(name);
        H5Dclose(dataset);
        return 0;
    };
    vector<string> ids;
    Group group = Group(file->openGroup("/trained"));
    herr_t idx = H5Literate(group.getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, &ids);

    auto& trained = data.trained;
    for (auto& id : ids)
    {
        vector<double> aux_list;

        DataSet dataset = DataSet(group.openDataSet(id));
        auto   datatype   = dataset.getDataType();
        DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims, NULL);

        size_t test_size = 1;
        for (int i = 0; i < rank; ++i)
            test_size *= dims[i];
        aux_list.resize(test_size);
        dataset.read(&aux_list[0], datatype);

        trained.insert({id, aux_list});
    }
        
    cout<<"trained data size: "<<trained.size()<<endl;

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

    // read celltypes
    readCelltype(file, data);

    // read trained data
    readTrained(file, data);

    // clear resources
    delete file;

    return true;
}

