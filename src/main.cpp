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
    if (argc < 4)
    {
        cerr << "enter <input.h5> <ref h5> <qry h5> [mode]" << endl;
        exit(-1);
    }
    string filename(argv[1]);
    string ref_h5(argv[2]);
    string qry_h5(argv[3]);
    int    mod = 0;
    if (argc == 5)
    {
        mod = stoi(argv[4]);
    }
    Pipeline pipeline = Pipeline();
    pipeline.train(filename, ref_h5, qry_h5);
    pipeline.score(mod);
    pipeline.finetune(mod);

    return 0;
}