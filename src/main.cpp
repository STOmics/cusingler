/* Copyright (C) BGI-Reasearch - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by STOmics development team P_stomics_dev@genomics.cn, 2023
*/

#include "io.h"
#include "types.h"

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
    InputData data;
    if (!readInput(filename, data))
        cerr<<"failed reading input h5 file."<<endl;
    else
        cout<<"success reading input h5 file."<<endl;

    return 0;
}