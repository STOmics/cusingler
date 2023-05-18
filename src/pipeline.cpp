/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "pipeline.h"
#include "cusingler.cuh"
#include "timer.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <thread>
#include <fstream>
using namespace std;

Pipeline::Pipeline(string ref_file, string qry_file, string stat_file, int rank_mode) :
    ref_file(ref_file), qry_file(qry_file), stat_file(stat_file), rank_mode(rank_mode)
{
}

bool Pipeline::train()
{
    cout << "start training ref data." << endl;
    Timer timer;

    data_parser = new DataParser(ref_file, qry_file);
    data_parser->findIntersectionGenes();
    data_parser->loadRefData();
    data_parser->trainData();
    data_parser->loadQryData();
    data_parser->preprocess();
    cout<<"train data cost time(s): "<<timer.toc()<<endl;

    return true;
}

bool Pipeline::score()
{
    cout << "start get labels." << endl;
    Timer timer;

    data_parser->generateDenseMatrix(0);
    auto& raw_data = data_parser->raw_data;
    raw_data.labels.clear();
    raw_data.labels.resize(raw_data.ct_num * raw_data.qry_height, 0);
    
    cells = raw_data.qry_cellnames;
   
    init();
    copyin_score(raw_data);
    
    auto first_label_index = get_label(raw_data, rank_mode);
    for (auto& i : first_label_index)
        first_labels.push_back(raw_data.celltypes[i]);

    destroy_score();

    cout<<"score data cost time(s): "<<timer.toc()<<endl;

    return true;
}


bool Pipeline::finetune()
{
    cout << "start finetune." << endl;
    Timer timer("ms");

    data_parser->generateDenseMatrix(1);    

    auto& raw_data = data_parser->raw_data;
   
    init();
    copyin(raw_data, raw_data.ctidx, raw_data.ctdiff, raw_data.ctdidx, raw_data.ref, raw_data.qry);
    auto  res = cufinetune(rank_mode);
    for (auto& c : res)
        final_labels.push_back(raw_data.celltypes[c]);

    destroy();
    cout << "finetune cost time(ms): " << timer.toc() << endl;

    return true;
}

bool Pipeline::dump()
{
    ofstream ofs(stat_file);
    ofs <<"cell\tfirstLabel\tfinalLabel\n";
    for (int i = 0; i < final_labels.size(); ++i)
        ofs << cells[i]<<"\t"<<first_labels[i]<<"\t"<<final_labels[i]<<endl;
    ofs.close();
    return true;
}