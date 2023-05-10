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
    if (argc != 2 && argc != 3)
    {
        cerr << "enter <input.h5>" << endl;
        exit(-1);
    }
    string filename(argv[1]);
    int    mod = 0;
    if (argc == 3)
    {
        mod = stoi(argv[2]);
    }
    Pipeline pipeline = Pipeline(filename);
    pipeline.preprocess();
    pipeline.score_data(mod);
    pipeline.work(mod);

    return 0;
}