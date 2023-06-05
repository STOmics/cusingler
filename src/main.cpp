/* Copyright (C) BGI-Reasearch - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by STOmics development team P_stomics_dev@genomics.cn, 2023
 */

#include "pipeline.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <thread>
using namespace std;

#include <CLI/CLI.hpp>

constexpr auto APP_NAME    = "cusingler";
constexpr auto APP_VERSION = "1.0";

int main(int argc, char** argv)
{
    // Parse the command line parameters.
    CLI::App app{ string(APP_NAME) + ": Accelerate singleR with GPU." };
    app.footer(string(APP_NAME) + " version: " + APP_VERSION);
    app.get_formatter()->column_width(40);

    // Required parameters
    string ref_h5, qry_h5, stat_file;
    app.add_option("-r,--ref", ref_h5, "Reference h5 file")
        ->check(CLI::ExistingFile)
        ->required();
    app.add_option("-q,--qry", qry_h5, "Query h5 file")
        ->check(CLI::ExistingFile)
        ->required();
    app.add_option("-o,--out", stat_file, "Output stat file, seperated by tab")
        ->required();

    // Optional parameters
    int cores = std::thread::hardware_concurrency();
    app.add_option("-c", cores, "CPU core number, default detect")
        ->check(CLI::PositiveNumber);
    int gpuid = 0;
    app.add_option("-g", gpuid, "Set gpu id, default 0")->check(CLI::NonNegativeNumber);

    CLI11_PARSE(app, argc, argv);

    Timer    timer;
    Pipeline pipeline = Pipeline(ref_h5, qry_h5, stat_file, cores, gpuid);
    pipeline.train();
    pipeline.score();
    pipeline.finetune();
    pipeline.dump();
    cout << "Total cost time(s): " << timer.toc() << endl;

    return 0;
}