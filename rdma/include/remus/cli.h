#pragma once

#include <cstdio>
#include <format>
#include <libgen.h>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace remus {

/// @brief A struct to classify command-line arguments
/// @details
/// A tuple that describes a command-line argument and its value. Supported 
/// types are uint64_t, double, bool, and std::string. Note that only one flag
/// per option is supported, and it must begin with '-'.  Note, too, that a
/// command-line argument is optional if a default value is provided.
struct Arg {
  /// A variant holding the four supported types of arg values
  using value_t = std::variant<uint64_t, double, std::string, bool>;

  /// The possible types of an argument.  These help with disambiguating value_t
  enum ArgValType { U64, F64, STR, BOOL };

  std::string flag;                 // The flag (e.g., -h or --help) for the arg
  std::string description;          // A description for usage()
  ArgValType type;                  // The type of value in this arg
  std::optional<value_t> value;     // The value of this arg
  std::vector<std::string> options; // The options, if it's a string enum
};

/// Construct an optional command-line arg of type std::string
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
inline Arg STR_ARG_OPT(std::string flag, std::string desc,
                       std::string def_val) {
  return {flag, desc, Arg::STR, def_val, {}};
}

/// Construct a required command-line arg of type std::string
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
///
/// @return An Arg for this command-line arg
inline Arg STR_ARG(std::string flag, std ::string desc) {
  return {flag, desc, Arg::STR, {}, {}};
}

/// Construct an optional command-line arg that is limited to a fixed-length
/// enumeration of strings
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
/// @param options  The valid options
///
/// @return An Arg for this command-line arg
inline Arg ENUM_ARG_OPT(std::string flag, std::string desc, std::string def_val,
                        std::vector<std::string> options) {
  return {flag, desc, Arg::STR, def_val, options};
}

/// Construct a required command-line arg that is limited to a fixed-length
/// enumeration of strings
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param options  The valid options
///
/// @return An Arg for this command-line arg
inline Arg ENUM_ARG(std::string flag, std ::string desc,
                    std::vector<std::string> options) {
  return {flag, desc, Arg::STR, std::nullopt, options};
}

/// Construct an optional command-line arg that is a bool.  It will default to
/// false, because bool flags don't ever take an argument
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
inline Arg BOOL_ARG_OPT(std::string flag, std ::string desc) {
  return {flag, desc, Arg::BOOL, false, {}};
}

/// Construct an optional command-line arg of type uint64_t
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
inline Arg U64_ARG_OPT(std::string flag, std ::string desc, uint64_t def_val) {
  return {flag, desc, Arg::U64, def_val, {}};
}

/// Construct a required command-line arg of type uint64_t
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
///
/// @return An Arg for this command-line arg
inline Arg U64_ARG(std::string flag, std::string desc) {
  return {flag, desc, Arg::U64, {}, {}};
}

/// Construct an optional command-line arg of type double
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
inline Arg F64_ARG_OPT(std::string flag, std::string desc, double def_val) {
  return {flag, desc, Arg::F64, def_val, {}};
}

/// Construct a required command-line arg of type double
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
///
/// @return An Arg for this command-line arg
inline Arg F64_ARG(std::string flag, std ::string desc) {
  return {flag, desc, Arg::F64, {}, {}};
}

/// @brief A collection of Args, and associated methods for working with them
/// @details 
/// A collection of Args, and associated methods for working with them. Note
/// that ArgMap is trivially constructed, but you probably want to use
/// `import_args` once (or more times) to populate it.
class ArgMap {
  /// A mapping from Arg.flag to an Arg, representing all supported args
  std::map<std::string, Arg> args;

  /// The name of the program being run.  This also serves as a flag to indicate
  /// that parse has been run.
  std::string program_name = "";

public:
  /// Merge a bunch of Arg objects into the map of supported args.  Fail if `in`
  /// includes keys that have already been imported.
  ///
  /// @param in The args to merge into the ArgMap
  void import(const std::initializer_list<Arg> &in) {
    using namespace std;
    if (program_name != "") {
      throw runtime_error("Error: cannot call import_args() after parse()");
    }
    for (auto c : in) {
      if ((c.flag.length() < 2) || (*c.flag.begin() != '-')) {
        throw runtime_error(format("Error: invalid flag `{}`", c.flag));
      }
      if (args.find(c.flag) != args.end()) {
        throw runtime_error(format("Error: duplicate flag `{}`", c.flag));
      }
      args.insert({c.flag, c});
    }
  }

  /// Try to process the command-line args, according to the Arg objects that
  /// have been imported into this ArgMap.  Note that we currently only support
  /// named arguments, unlike getopt(), which moves all unnamed arguments to the
  /// end of argv.  Fail if any required arg was omitted.
  ///
  /// @param argc The number of command-line args
  /// @param argv The array of command-line args
  void parse(int argc, char **argv) {
    using namespace std;
    // Ensure this wasn't called more than once
    if (program_name != "") {
      throw runtime_error("Error: parse() should only be called once!");
    }
    program_name = basename(argv[0]);

    int curr = 1;
    while (curr < argc) {
      string ca = argv[curr];
      auto arg = args.find(ca);
      if (arg == args.end()) {
        usage();
        throw runtime_error(format("Error: unrecognized argument `{}`", ca));
      }
      // Handle bools first, because they don't take a value
      if (arg->second.type == Arg::BOOL) {
        arg->second.value = true;
        ++curr;
        continue;
      }
      // Fail if we're at the end, and there's no value
      if (curr == (argc - 1)) {
        usage();
        throw runtime_error(
            format("Error: argument `{}` requires a value", ca));
      }
      // Fail if the next thing isn't a value
      string next = argv[curr + 1];
      if (*next.begin() == '-') {
        usage();
        throw runtime_error(
            format("Error: argument `{}` requires a value", ca));
      }

      // Now we can parse the value and advance
      if (arg->second.type == Arg::U64) {
        arg->second.value = stoull(next);
      } else if (arg->second.type == Arg::F64) {
        arg->second.value = stod(next);
      } else if (arg->second.type == Arg::STR) {
        arg->second.value = next;
        if (arg->second.options.size() > 0) {
          bool match = false;
          for (auto i : arg->second.options) {
            match = match || (i == next);
          }
          if (!match) {
            usage();
            throw runtime_error(
                format("Error: invalid value for argument `{}`", ca));
          }
        }
      }
      // TODO: I don't see where we handle Bool args?
      curr += 2;
    }

    // Verify that no required args were skipped
    for (auto a : args) {
      if (!a.second.value) {
        usage();
        throw runtime_error(format("Error: `{}` is required", a.second.flag));
      }
    }
  }

  /// Print a usage message
  void usage() {
    using namespace std;
    printf("%s", format("{}\n", program_name).c_str());
    for (auto c : args)
      printf("%s", format("  {}: {}\n", c.first, c.second.description).c_str());
  }

  /// Print a message describing the current state of the command-line args
  void report_config() {
    using namespace std;
    printf("%s", format("{} (", program_name).c_str());
    for (auto c : args)
      printf("%s", format("{} ", c.second.flag).c_str());
    printf(")");
    for (auto c : args) {
      printf(", ");
      if (c.second.type == Arg::BOOL)
        printf("%s", get<bool>(c.second.value.value()) ? "true" : "false");
      else if (c.second.type == Arg::U64)
        printf("%lu", get<uint64_t>(c.second.value.value()));
      else if (c.second.type == Arg::F64)
        printf("%lf", get<double>(c.second.value.value()));
      else if (c.second.type == Arg::STR)
        printf("%s", get<string>(c.second.value.value()).c_str());
    }
    printf("\n");
  }

  /// Get an argument's value as a boolean
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as a bool
  bool bget(std::string flag) {
    return std::get<bool>(args.find(flag)->second.value.value());
  }

  /// Get an argument's value as an uint64_t
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as an uint64_t
  uint64_t uget(std::string flag) {
    return std::get<uint64_t>(args.find(flag)->second.value.value());
  }

  /// Get an argument's value as a double
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as a double
  double fget(std::string flag) {
    return std::get<double>(args.find(flag)->second.value.value());
  }

  /// Get an argument's value as a string
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as a string
  std::string sget(std::string flag) {
    return std::get<std::string>(args.find(flag)->second.value.value());
  }
};
} // namespace remus
