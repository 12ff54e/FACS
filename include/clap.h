/*
 * Command Line Argument Parser
 */

#ifndef ZQ_CLAP
#define ZQ_CLAP

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define TYPE_LIST()        \
    PROCESS_TYPE(bool)     \
    PROCESS_TYPE(int)      \
    PROCESS_TYPE(unsigned) \
    PROCESS_TYPE(double)   \
    PROCESS_TYPE(float)    \
    PROCESS_TYPE(string)

enum class CLAP_TYPE_CODE {
    _help_t,
#define PROCESS_TYPE(TYPE) TYPE##_t,
    TYPE_LIST()
#undef PROCESS_TYPE
};

struct CLAP_OptionConfig {
    std::size_t offset;
    CLAP_TYPE_CODE type_code;
    std::string short_name;
    std::string description;

    bool has_short_name() const { return !short_name.empty(); }
};

template <typename T>
struct CLAP {
    using base = T;
    using string = std::string;
    using option_container_type =
        std::unordered_map<std::string, CLAP_OptionConfig>;

    static auto& get_options() {
        static option_container_type options{};
        return options;
    }

    static auto& get_short_name_map() {
        static std::unordered_map<std::string,
                                  option_container_type::const_iterator>
            short_name_map;
        return short_name_map;
    }

    static auto& get_arguments() {
        static std::vector<std::pair<std::size_t, CLAP_TYPE_CODE>> arguments;
        return arguments;
    }

    static auto& get_usage() {
        static std::string usage;
        return usage;
    }

    static auto& get_description() {
        static std::string description;
        return description;
    }

    static std::string process_option(std::string opt, bool short_opt = false) {
        bool valid_name = true;
        if ((valid_name = !opt.empty()) && opt[0] == '-') {
            if (short_opt) { return opt; }
            if (opt.size() > 1 && opt[1] == '-') {
                if (!short_opt) { return opt; }
            }
            valid_name = false;
        }
        if (!valid_name) {
            std::ostringstream oss;
            oss << "Invalid " << (short_opt ? "short" : "long")
                << " option definition: '" << opt << "'\n";
            throw std::invalid_argument(oss.str());
        }

        return (short_opt ? "-" : "--") + opt;
    }

    template <typename Arg>
    static CLAP_TYPE_CODE get_type_code() {
#define PROCESS_TYPE(TYPE) \
    if (std::is_same<Arg, TYPE>::value) { return CLAP_TYPE_CODE::TYPE##_t; }
        TYPE_LIST()
#undef PROCESS_TYPE
        throw std::invalid_argument("Type not supported yet.");
    }

    static void assign_value(char* adr,
                             CLAP_TYPE_CODE type_code,
                             const std::string& raw_value) {
        std::istringstream iss(raw_value);
#define PROCESS_TYPE(TYPE)                    \
    case (CLAP_TYPE_CODE::TYPE##_t): {        \
        iss >> *reinterpret_cast<TYPE*>(adr); \
        break;                                \
    }

        switch (type_code) {
            TYPE_LIST()
            default:
                assert(0 && "Unreachable");
        }

#undef PROCESS_TYPE
    }

    static void parse_input(T& input, int argc, char** argv) {
        std::size_t argument_idx = 0;
        bool argument_mode = false;
        for (int i = 1; i < argc; ++i) {
            std::string option_or_arg{argv[i]};
            if (option_or_arg.compare("--") == 0) {
                argument_mode = true;
                continue;
            }
            if (!argument_mode && option_or_arg[0] == '-') {
                const auto& opts = get_options();
                std::size_t offset{};
                CLAP_TYPE_CODE type_code{};
                auto iter = opts.cbegin();
                std::string raw_value;

                bool invalid_option = false;
                bool has_value = false;

                // parse option name
                if (option_or_arg[1] == '-') {
                    // long option
                    std::size_t equal_sign_idx = 0;
                    if ((equal_sign_idx = option_or_arg.find('=')) !=
                        std::string::npos) {
                        iter =
                            opts.find(option_or_arg.substr(0, equal_sign_idx));
                        raw_value = option_or_arg.substr(equal_sign_idx + 1,
                                                         std::string::npos);
                        has_value = true;
                    } else {
                        iter = opts.find(option_or_arg);
                    }
                    invalid_option = iter == opts.end();
                } else {
                    // short option
                    auto& short_opts = get_short_name_map();
                    auto iter_short = short_opts.find(option_or_arg);
                    if (!(invalid_option = iter_short == short_opts.end())) {
                        iter = iter_short->second;
                    }
                }
                if (invalid_option) {
                    std::ostringstream oss;
                    oss << argv[0] << ": unrecognized option '" << option_or_arg
                        << "'\nTry '" << argv[0]
                        << " --help' for more information.\n";
                    std::size_t min_dist = 10;
                    auto min_iter = opts.cbegin();
                    for (auto it = opts.begin(); it != opts.end(); ++it) {
                        auto dist = edit_distance(it->first, option_or_arg);
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_iter = it;
                        }
                    }
                    if (min_dist < 3) {
                        oss << "\n    Do you mean '" << min_iter->first
                            << "'?\n";
                    }
                    throw std::invalid_argument(oss.str());
                } else {
                    offset = iter->second.offset;
                    type_code = iter->second.type_code;
                }

                // is it help?
                if (type_code == CLAP_TYPE_CODE::_help_t) {
                    print_help(argv[0]);
                    std::exit(0);
                }

                // parse option value
                if (!has_value) {
                    if (type_code == CLAP_TYPE_CODE::bool_t) {
                        raw_value = "1";
                    } else if (i + 1 == argc) {
                        std::ostringstream oss;
                        oss << argv[0] << ": missing value of option '"
                            << option_or_arg << "'\nTry '" << argv[0]
                            << " --help' for more information.\n";
                        throw std::invalid_argument(oss.str());
                    } else {
                        raw_value = argv[++i];
                    }
                }
                assign_value(reinterpret_cast<char*>(&input) + offset,
                             type_code, raw_value);
            } else {
                // it's an argument
                if (argument_idx == get_arguments().size()) {
                    std::ostringstream oss;
                    oss << argv[0] << ": too many arguments\n"
                        << "Try '" << argv[0]
                        << " --help' for more information.\n";
                    throw std::invalid_argument(oss.str());
                }
                auto [offset, type_code] = get_arguments()[argument_idx];
                assign_value(reinterpret_cast<char*>(&input) + offset,
                             type_code, argv[i]);
                ++argument_idx;
            }
        }
        if (argument_idx < get_arguments().size()) {
            std::ostringstream oss;
            oss << argv[0] << ": too few arguments\n"
                << "Try '" << argv[0] << " --help' for more information.\n";
            throw std::invalid_argument(oss.str());
        }
    }

    static void print_help(const char* program_name) {
        std::vector<option_container_type::const_iterator> iters;
        for (auto it = get_options().begin(); it != get_options().end(); ++it) {
            iters.push_back(it);
        }
        std::sort(iters.begin(), iters.end(),
                  [](auto it1, auto it2) { return it1->first < it2->first; });
        auto print_space = [](std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) { std::cout << ' '; }
        };
        auto print_multline = [&print_space](const std::string& str,
                                             std::size_t width,
                                             std::size_t indent) {
            std::size_t last_space_idx = 0;
            std::size_t current_space_idx = 0;
            std::size_t line_begin_idx = 0;
            while (current_space_idx < std::string::npos) {
                if (line_begin_idx > 0) { print_space(indent); }
                last_space_idx = line_begin_idx;
                while ((current_space_idx = str.find(' ', last_space_idx + 1)) <
                       line_begin_idx + width) {
                    last_space_idx = current_space_idx;
                }
                if (current_space_idx == std::string::npos) {
                    if (str.size() - line_begin_idx > width) {
                        // last word at new line
                        current_space_idx = str.size();
                    } else {
                        last_space_idx = str.size();
                    }
                }
                std::cout.write(str.data() + line_begin_idx,
                                static_cast<std::streamsize>(last_space_idx -
                                                             line_begin_idx))
                    << '\n';
                line_begin_idx = last_space_idx + 1;
            }
        };

        std::cout << "Usage: " << program_name << " " << get_usage() << '\n';
        std::cout << get_description() << "\n\n";
        for (auto it : iters) {
            std::size_t col = 0;
            if (it->second.has_short_name()) {
                print_space(2);
                std::cout << it->second.short_name << ", ";
                col = 4 + it->second.short_name.size();
            } else {
                print_space(6);
                col = 6;
            }
            std::cout << it->first;  // long name
            col += it->first.size();
            if (col > 28) {
                std::cout << '\n';
                print_space(30);
            } else {
                print_space(30 - col);
            }
            print_multline(it->second.description, 50, 32);
        }
    }

    static void report_repeat_definition(std::string option_name) {
#ifdef CLAP_DEBUG
        std::ostringstream oss;
        oss << "You can not define option '" << option_name << "' twice!\n";
        throw std::invalid_argument(oss.str());
#else
        (void)option_name;
#endif
    }

    static std::size_t edit_distance(const std::string& str1,
                                     const std::string& str2) {
        std::size_t m = str1.size() + 1;
        std::size_t n = str2.size() + 1;
        std::vector<std::size_t> sub_dist(m * n);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                sub_dist[i * n + j] =
                    i == 0   ? j
                    : j == 0 ? i
                    : str1[i - 1] == str2[j - 1]
                        ? sub_dist[(i - 1) * n + j - 1]
                        : 1 + std::min({sub_dist[(i - 1) * n + j - 1],
                                        sub_dist[(i - 1) * n + j],
                                        sub_dist[i * n + j - 1]});
            }
        }
        return sub_dist[m * n - 1];
    }

#undef TYPE_LIST
};

#define CLAP_BEGIN(NAME)                        \
    struct _clap_##NAME##_ : NAME, CLAP<NAME> { \
        using base = NAME;                      \
        static void define_parameters() {
#define CLAP_END(NAME)                                                 \
    {                                                                  \
        auto result = get_options().emplace(                           \
            "--help",                                                  \
            CLAP_OptionConfig{0, CLAP_TYPE_CODE::_help_t, "",          \
                              "display this help message and exit"});  \
        if (get_short_name_map().emplace("-h", result.first).second) { \
            result.first->second.short_name = "-h";                    \
        }                                                              \
    }                                                                  \
    }                                                                  \
    }                                                                  \
    ;                                                                  \
    try {                                                              \
        _clap_##NAME##_::define_parameters();                          \
    } catch (std::exception & e) {                                     \
        std::cerr << e.what();                                         \
        return EINVAL;                                                 \
    }

#define CLAP_REGISTER_OPT_MINIMAL(NAME)                                      \
    {                                                                        \
        auto result = get_options().emplace(                                 \
            "--" #NAME, CLAP_OptionConfig{offsetof(base, NAME),              \
                                          get_type_code<decltype(NAME)>()}); \
        if (!result.second) { report_repeat_definition(OPTION_NAME); }       \
    }

#define CLAP_REGISTER_OPT_LONG(NAME, OPTION_NAME)                             \
    {                                                                         \
        auto option_name = process_option(OPTION_NAME);                       \
        auto result = get_options().emplace(                                  \
            option_name, CLAP_OptionConfig{offsetof(base, NAME),              \
                                           get_type_code<decltype(NAME)>()}); \
        if (!result.second) { report_repeat_definition(option_name); }        \
    }

#define CLAP_REGISTER_OPT_LONG_SHORT(NAME, OPTION_NAME, OPTION_NAME_SHORT) \
    {                                                                      \
        auto option_name = process_option(OPTION_NAME);                    \
        auto option_name_short = process_option(OPTION_NAME_SHORT, true);  \
        auto result = get_options().emplace(                               \
            option_name,                                                   \
            CLAP_OptionConfig{offsetof(base, NAME),                        \
                              get_type_code<decltype(NAME)>(),             \
                              process_option(option_name_short, false)});  \
        if (!result.second) { report_repeat_definition(option_name); }     \
        get_short_name_map().emplace(option_name_short, result.first);     \
    }

#define CLAP_REGISTER_OPT_DESC(NAME, DESC)                                 \
    {                                                                      \
        auto result = get_options().emplace(                               \
            "--" #NAME,                                                    \
            CLAP_OptionConfig{offsetof(base, NAME),                        \
                              get_type_code<decltype(NAME)>(), "", DESC}); \
        if (!result.second) { report_repeat_definition("--" #NAME); }      \
    }

#define CLAP_REGISTER_OPT_LONG_DESC(NAME, OPTION_NAME, DESC)               \
    {                                                                      \
        auto option_name = process_option(OPTION_NAME);                    \
        auto result = get_options().emplace(                               \
            option_name,                                                   \
            CLAP_OptionConfig{offsetof(base, NAME),                        \
                              get_type_code<decltype(NAME)>(), "", DESC}); \
        if (!result.second) { report_repeat_definition(option_name); }     \
    }

#define CLAP_REGISTER_OPT_FULL(NAME, OPTION_NAME, OPTION_NAME_SHORT, DESC)  \
    {                                                                       \
        auto option_name = process_option(OPTION_NAME);                     \
        auto option_name_short = process_option(OPTION_NAME_SHORT, true);   \
        auto result = get_options().emplace(                                \
            option_name, CLAP_OptionConfig{offsetof(base, NAME),            \
                                           get_type_code<decltype(NAME)>(), \
                                           option_name_short, DESC});       \
        if (!result.second) { report_repeat_definition(option_name); }      \
        get_short_name_map().emplace(option_name_short, result.first);      \
    }

// Macro overloading depending on argument number
// https://stackoverflow.com/a/11763277/7255197
#define CLAP_GET_MACRO_3(_1, _2, _3, NAME, ...) NAME
#define CLAP_REGISTER_OPTION(...)                                         \
    CLAP_GET_MACRO_3(__VA_ARGS__, CLAP_REGISTER_OPT_LONG_SHORT,           \
                     CLAP_REGISTER_OPT_LONG, CLAP_REGISTER_OPT_MINIMAL, ) \
    (__VA_ARGS__)

#define CLAP_GET_MACRO_4(_1, _2, _3, _4, NAME, ...) NAME
#define CLAP_REGISTER_OPTION_WITH_DESCRIPTION(...)                          \
    CLAP_GET_MACRO_4(__VA_ARGS__, CLAP_REGISTER_OPT_FULL,                   \
                     CLAP_REGISTER_OPT_LONG_DESC, CLAP_REGISTER_OPT_DESC, ) \
    (__VA_ARGS__)

#define CLAP_REGISTER_ARG(NAME)                                        \
    {                                                                  \
        get_arguments().emplace_back(offsetof(base, NAME),             \
                                     get_type_code<decltype(NAME)>()); \
    }

#define CLAP_ADD_USAGE(USAGE) get_usage() = USAGE;
#define CLAP_ADD_DESCRIPTION(DESC) get_description() = DESC;

#endif  // ZQ_CLAP
