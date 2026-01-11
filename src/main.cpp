#include "cxxopts.hpp"
#include "str_helpers.h"
#include "netlist/netlist.hpp"
#include "rlc2ss.h"

#include <filesystem>
#include <chrono>

int main(int argc, char** argv) {
    cxxopts::Options options("Generate C++ state-space matrices from given netlist");
    options.allow_unrecognised_options();
    // clang-format off
    options.add_options()
        ("h,help", "Show help and exit")
        ("v,verbose", "Enable verbose output");
    // clang-format on
    cxxopts::ParseResult parsed_options;
    bool verbose = false;
    try {
        parsed_options = options.parse(argc, argv);
        if (parsed_options.count("verbose")) {
            verbose = true;
        }
    } catch (const cxxopts::OptionException& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }
    auto unmatched_options = parsed_options.unmatched();
    if (parsed_options.count("help") || unmatched_options.size() != 1) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string netlist_path = unmatched_options[0];
    try {
        std::expected<std::string, std::string> file_content = rlc2ss::str::readFile(netlist_path);
        if (!file_content) {
            throw std::runtime_error("Failed to read netlist file: " + file_content.error());
        }
        std::vector<std::string> netlist_lines = rlc2ss::collectNetlistLines(*file_content);
        std::vector<std::string> switches = rlc2ss::extractSwitches(netlist_lines);
        for (int combination = 0; combination < (1 << switches.size()); ++combination) {
            std::cout << combination << " ";
            auto t_start = std::chrono::steady_clock::now();
            rlc2ss::StateSpaceMatrices output = rlc2ss::formStateSpaceMatrices(*file_content, combination, {}, verbose);
            auto t_end = std::chrono::steady_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
            std::cout << "formStateSpaceMatrices took " << elapsed_us << " ms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
