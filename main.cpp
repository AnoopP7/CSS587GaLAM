/*
 * test_main.cpp
 * CSS587 Design Project - GaLAM Testing
 * Implementation authors: Yu Dinh, Neha Kotwal, Anoop Prasad
 * Paper title: GaLAM: Two-Stage Outlier Detection Algorithm
 * Paper authors: X. Lu, Z. Yan, Z. Fan
 *
 * This file contains the main function to run GaLAM tests over a dataset.
 * Purpose:
 * - Provide a command-line interface for testing the GaLAM algorithm
 * - Facilitate the evaluation of different feature detectors and matching methods
 * - Generate and save test results for analysis
 */

#include "match_test.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <data_path> [output.csv]\n";
        return 1;
    }

    MatchTest tester(
        {MatchTest::Detector::SIFT},
        {MatchTest::Method::NN_RT, MatchTest::Method::RANSAC, MatchTest::Method::GALAM});

    tester.runTests(argv[1], argc >= 3 ? argv[2] : "./output/results.csv");
    return 0;
}