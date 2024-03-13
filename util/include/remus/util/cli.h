// A generic facility for working with command-line arguments.  The supported
// argument types are int64_t, double, bool, and std::string.

#pragma once

#include <iostream>
#include <libgen.h>
#include <map>
#include <optional>
#include <string>
#include <variant>

namespace remus::util {

/// @brief  A tuple that describes a command-line argument and its value.
///         Supported types are int64_t, double, bool, and std::string.
///         Note that only one flag per option is supported, and it must begin
///         with '-'.  Note, too, that a command-line argument is optional
///         iff a default value is provided.
struct Arg {
  /// @brief  A variant holding the four supported types of arg values
  using value_t = std::variant<int64_t, double, std::string, bool>;

  /// @brief  The possible types of an argument.  These help with disambiguating
  ///         value_t
  enum ArgValType { I64, F64, STR, BOOL };

  std::string flag;             // The flag (e.g., -h or --help) for this arg
  const char *description;      // A description for usage()
  ArgValType type;              // The type of value in this arg
  std::optional<value_t> value; // The value of this arg
};

/// @brief  Construct an optional command-line arg of type std::string
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
Arg STR_ARG_OPT(std::string flag, const char *desc, std::string def_val) {
  return {flag, desc, Arg::STR, def_val};
}

/// @brief  Construct a required command-line arg of type std::string
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
///
/// @return An Arg for this command-line arg
Arg STR_ARG(std::string flag, const char *desc) {
  return {flag, desc, Arg::STR, std::nullopt};
}

/// @brief  Construct an optional command-line arg that is a bool.  It will
///         default to false, because bool flags don't ever take an argument
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
Arg BOOL_ARG_OPT(std::string flag, const char *desc) {
  return {flag, desc, Arg::BOOL, false};
}

/// @brief  Construct an optional command-line arg of type int64_t
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
Arg I64_ARG_OPT(std::string flag, const char *desc, int64_t def_val) {
  return {flag, desc, Arg::I64, def_val};
}

/// @brief  Construct a required command-line arg of type int64_t
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
///
/// @return An Arg for this command-line arg
Arg I64_ARG(std::string flag, const char *desc) {
  return {flag, desc, Arg::I64, std::nullopt};
}

/// @brief  Construct an optional command-line arg of type double
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
/// @param def_val  The default value
///
/// @return An Arg for this command-line arg
Arg F64_ARG_OPT(std::string flag, const char *desc, double def_val) {
  return {flag, desc, Arg::F64, def_val};
}

/// @brief  Construct a required command-line arg of type double
///
/// @param flag     The flag (e.g., -h) for this argument
/// @param desc     A description (for help)
///
/// @return An Arg for this command-line arg
Arg F64_ARG(std::string flag, const char *desc) {
  return {flag, desc, Arg::F64, std::nullopt};
}

/// @brief A collection of Args, and associated methods for working with them
class ArgMap {
  /// @brief  A mapping from Arg.flag to an Arg, representing all supported args
  std::map<std::string, Arg> args;

  /// @brief  The name of the program being run.  This also serves as a flag to
  ///         indicate that parse_args has been run.
  std::string program_name = "";

public:
  /// @brief  Merge a bunch of Arg objects into the map of supported args.  Fail
  ///         if `in` includes keys that have already been imported.
  ///
  /// @param in The args to merge into the ArgMap
  ///
  /// @return A string error, or {}
  [[nodiscard]] std::optional<std::string>
  import_args(const std::initializer_list<Arg> &in) {
    if (program_name != "")
      return "Error: cannot call import_args() after parse_args()";
    for (auto c : in) {
      if ((c.flag.length() < 2) || (*c.flag.begin() != '-'))
        return std::string("Error: invalid flag `") + c.flag + "`";
      if (args.find(c.flag) != args.end())
        return std::string("Error: duplicate flag `") + c.flag + "`";
      args.insert({c.flag, c});
    }
    return {};
  }

  /// @brief  Try to process the command-line args, according to the Arg objects
  ///         that have been imported into this ArgMap.  Note that we currently
  ///         only support named arguments, unlike getopt(), which moves all
  ///         unnamed arguments to the end of argv.  Fail if any required arg
  ///         was omitted.
  ///
  /// @param argc The number of command-line args
  /// @param argv The array of command-line args
  ///
  /// @return A string error, or {}
  std::optional<std::string> parse_args(int argc, char **argv) {
    // NB:  Disable future calls to parse_args() and import_args()
    if (program_name != "")
      return "Error: parse_args() should only be called once!";
    program_name = basename(argv[0]);

    int curr = 1;
    while (curr < argc) {
      std::string ca = argv[curr];
      auto arg = args.find(ca);
      if (arg == args.end())
        return std::string("Error: unrecognized argument `") + ca + "`";
      // Handle bools first, because they don't take a value
      if (arg->second.type == Arg::BOOL) {
        arg->second.value = true;
        ++curr;
        continue;
      }
      // Fail if we're at the end, and there's no value
      if (curr == (argc - 1))
        return std::string("Error: argument `") + ca + "` requires a value";
      // Fail if the next thing isn't a value
      std::string next = argv[curr + 1];
      if (*next.begin() == '-')
        return std::string("Error: argument `") + ca + "` requires a value";

      // Now we can parse the value and advance
      if (arg->second.type == Arg::I64)
        arg->second.value = std::stoi(next);
      else if (arg->second.type == Arg::F64)
        arg->second.value = std::stod(next);
      else if (arg->second.type == Arg::STR)
        arg->second.value = next;
      curr += 2;
    }

    // Verify that no required args were skipped
    for (auto a : args) {
      if (!a.second.value)
        return std::string("Error: `") + a.second.flag + "` is required";
    }

    return {};
  }

  /// @brief  Print a usage message
  void usage() {
    std::cout << program_name << "\n";
    for (auto c : args)
      std::cout << "  " << c.first << ": " << c.second.description << "\n";
  }

  /// @brief  Print a message describing the current state of the command-line
  ///         args
  void report_config() {
    std::cout << program_name;
    std::cout << " (";
    for (auto c : args)
      std::cout << c.second.flag << " ";
    std::cout << ")";
    for (auto c : args) {
      std::cout << ", ";
      if (c.second.type == Arg::BOOL)
        std::cout << (std::get<bool>(c.second.value.value()) ? "true"
                                                             : "false");
      else if (c.second.type == Arg::I64)
        std::cout << std::get<int64_t>(c.second.value.value());
      else if (c.second.type == Arg::F64)
        std::cout << std::get<double>(c.second.value.value());
      else if (c.second.type == Arg::STR)
        std::cout << std::get<std::string>(c.second.value.value());
    }
  }

  /// @brief  Get an argument's value as a boolean
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as a bool
  bool bget(std::string flag) {
    return std::get<bool>(args.find(flag)->second.value.value());
  }

  /// @brief  Get an argument's value as an int64_t
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as an int64_t
  int64_t iget(std::string flag) {
    return std::get<int64_t>(args.find(flag)->second.value.value());
  }

  /// @brief  Get an argument's value as a double
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as a double
  double fget(std::string flag) {
    return std::get<double>(args.find(flag)->second.value.value());
  }

  /// @brief  Get an argument's value as a string
  ///
  /// @param flag The flag to look up
  ///
  /// @return The value, as a string
  std::string sget(std::string flag) {
    return std::get<std::string>(args.find(flag)->second.value.value());
  }
};
} // namespace remus::util
