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

bool readInput(string& filename, InputData& data)
{
    // open h5 file handle
    H5File* file = new H5File(filename.c_str(), H5F_ACC_RDONLY);

    // read obs key and value, then construct map
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

    cout<<data.obs["mouse1_lib1.final_cell_0005"]<<endl;
    cout<<data.obs["mouse1_lib1.final_cell_0001"]<<endl;

    // clear resources
    delete file;

    return true;
}

// void parseGefTissueH5CPP(string tissueFile, unordered_set<uint64>& uniqCoors)
// {
//     vector<exp_t> data;
//     uint32 minx, miny;

//     try
//     {
//         H5File* file = new H5File(tissueFile.c_str(), H5F_ACC_RDONLY);
//         DataSet* dataset = new DataSet(file->openDataSet("/geneExp/bin1/expression"));

//         DataSpace dataspace = dataset->getSpace();
//         hsize_t dims[1];
//         dataspace.getSimpleExtentDims(dims, NULL);

//         Attribute attr = dataset->openAttribute("minX");
//         attr.read(PredType::NATIVE_UINT32, &minx);
//         attr = dataset->openAttribute("minY");
//         attr.read(PredType::NATIVE_UINT32, &miny);

//         CompType mtype(sizeof(exp_t));
//         mtype.insertMember("x", HOFFSET(exp_t, x), PredType::NATIVE_UINT32);
//         mtype.insertMember("y", HOFFSET(exp_t, y), PredType::NATIVE_UINT32);
//         mtype.insertMember("count", HOFFSET(exp_t, cnt), PredType::NATIVE_UINT32);

//         data.resize(dims[0]);
//         dataset->read(&data[0], mtype);
        
//         delete dataset;
//         delete file;
//     }
//     catch (...)
//     {
//         cerr<<"SAW-A80001 invalid tissueFile: "<<tissueFile<<endl;
//         throw std::runtime_error("invalid tissueFile!");
//     }

//     // Extract unique coordinates
//     for (auto& t : data)
//     {
//         uint64 coor = ((uint64)(t.x+minx) << 32) + t.y+miny;
//         uniqCoors.insert(coor);
//     }
// }
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

bool readInput(string& filename, InputData& data)
{
    // open h5 file handle
    H5File* file = new H5File(filename.c_str(), H5F_ACC_RDONLY);

    // read obs key and value, then construct map
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

    cout<<data.obs["mouse1_lib1.final_cell_0005"]<<endl;
    cout<<data.obs["mouse1_lib1.final_cell_0001"]<<endl;

    // clear resources
    delete file;

    return true;
}

