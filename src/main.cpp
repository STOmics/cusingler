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
    // TODO: set cpu number and gpu id from user input
    if (argc != 4)
    {
        cerr << "enter <ref h5> <qry h5> <result file>" << endl;
        exit(-1);
    }

    string ref_h5(argv[1]);
    string qry_h5(argv[2]);
    string stat_file(argv[3]);

    Timer    timer;
    Pipeline pipeline = Pipeline(ref_h5, qry_h5, stat_file);
    pipeline.train();
    pipeline.score();
    pipeline.finetune();
    pipeline.dump();
    cout << "Total cost time(s): " << timer.toc() << endl;

    return 0;
}