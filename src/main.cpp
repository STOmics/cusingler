/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "pipeline.h"
#include "timer.h"

#include <iostream>
#include <string>
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cerr << "enter <ref h5> <qry h5> <result file> [rank mode: 0 => bincount, 1 => count]" << endl;
        exit(-1);
    }

    string ref_h5(argv[1]);
    string qry_h5(argv[2]);
    string stat_file(argv[3]);
    int    rank_mode = 0;
    if (argc == 5)
    {
        rank_mode = stoi(argv[4]);
    }
    if(rank_mode == 0)
    {
        cout<<"rank data by bincount"<<endl;
    }
    else if (rank_mode == 1)
    {
        cout<<"rank data by count"<<endl;
    }
    else
    {
        cerr<<"invalid rank mode, please enter [0 => bincount, 1 => count]"<<endl;
        exit(-1);
    }

    Timer timer;
    Pipeline pipeline = Pipeline(ref_h5, qry_h5, stat_file, rank_mode);
    pipeline.train();
    pipeline.score();
    pipeline.finetune();
    pipeline.dump();
    cout<<"Total cost time(s): "<<timer.toc()<<endl;

    return 0;
}