/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "pipeline.h"

#include <iostream>
#include <string>
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cerr << "enter <ref h5> <qry h5> [rank mode: 0 => bincount, 1 => count]" << endl;
        exit(-1);
    }

    string ref_h5(argv[1]);
    string qry_h5(argv[2]);
    int    rank_mode = 0;
    if (argc == 4)
    {
        rank_mode = stoi(argv[3]);
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

    Pipeline pipeline = Pipeline(ref_h5, qry_h5, rank_mode);
    pipeline.train();
    pipeline.score();
    pipeline.finetune();

    return 0;
}