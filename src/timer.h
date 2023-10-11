
/*
 * File: Timer.h
 * Created Data: 2021-5-28
 * Author: tigeroses
 */

// Timer is a tool for timing code blocks.
//
// You can init the Timer with a timing unit, selected at:
// "s", "ms", "mics", the default unit is second. Use Timer::tic() to start a
// new timer, Timer::toc() to end the timer and return the duration. Examples:
//
//      Timer t1("s");
//      ...
//      std::cout<<"elapsed time(s): "<<t1.toc()<<std::endl;

#pragma once

#include <mutex>
#include <string>
#include <thread>

class Timer
{
public:
    Timer(std::string unit = "s") : unit(unit), start_time(mics_duration::zero())
    {
        if (unit == "s")
            div = 1e6;
        else if (unit == "ms")
            div = 1e3;
        else if (unit == "mics")
            div = 1;
        else
            div = 1e6;

        tic();
    }

    ~Timer() {}

    // Start a new timer
    inline void tic()
    {
        start_time = clock::now();
    }

    // Calculate the duration after last tic(), and start a new timer
    inline size_t toc()
    {
        // clang-format off
        size_t duration = std::chrono::duration_cast< mics_duration >(
                              clock::now() - start_time).count() / div;
        // clang-format on
        tic();

        return duration;
    }

private:
    std::string                                    unit;
    std::chrono::high_resolution_clock::time_point start_time;

    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::microseconds          mics_duration;
    unsigned int                               div;
};
