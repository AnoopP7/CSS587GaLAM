/*
 * test_main.cpp
 * CSS587 Design Project - GaLAM Testing
 */

#include "match_test.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data_path> [output.csv]\n";
        return 1;
    }

    MatchTest tester(
        {MatchTest::Detector::SIFT},
        {MatchTest::Method::NN_RT, MatchTest::Method::RANSAC, MatchTest::Method::GALAM}
    );

    tester.runTests(argv[1], argc >= 3 ? argv[2] : "results.csv");
    return 0;
}