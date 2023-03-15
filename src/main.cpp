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
    if (argc != 2)
    {
        cerr<<"enter <input.h5>"<<endl;
        exit(-1);
    }
    string filename(argv[1]);
    
    Pipeline pipeline = Pipeline(filename);
    pipeline.preprocess();
    pipeline.copytogpu();

    return 0;
}