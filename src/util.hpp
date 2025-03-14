#pragma once

#include <cstring>
#include <filesystem>
#include <format>
#include <list>
#include <map>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#ifdef DEBUG
#    include <cassert>
#    include <iostream>
#endif

inline std::string addIndent(std::string_view str, int indent = 1) {
    std::string indent_str;
    for (int i = 0; i < indent; i++) indent_str += "  ";
    bool indent_flag = 1;
    std::string result_str;
    for (auto i : str) {
        if (indent_flag) result_str += indent_str, indent_flag = 0;
        result_str += i;
        if (i == '\n') {
            indent_flag = 1;
        }
    }
    if (result_str.back() != '\n') result_str += '\n';
    return result_str;
}

inline std::string compressStr(std::string_view str) {
    std::string ret;
    char prev = ' ';
    for (auto i : str) {
        if (!isspace(i))
            ret += i;
        else if (!isspace(prev))
            ret += ' ';
        prev = i;
    }
    if (str.back() == '\n') ret += '\n';
    return ret;
}

inline std::string tryCompressStr(const std::string& str) {
    if (str.length() < 80)
        return compressStr(str);
    else
        return std::string(str);
}

std::string serialize(const auto& val);

template <std::input_iterator T> std::string toString(const T& begin, const T& end) {
    std::string str;
    for (T it = begin; it != end; it++) {
        str += serialize(*it) + ",\n";
    }
    if (!str.empty()) {
        str.pop_back(), str.pop_back();
    }
    return tryCompressStr(str);
}

template <typename T1, typename T2> std::string toString(const std::pair<T1, T2>& p) {
    return std::format("{}: {}", serialize(p.first), serialize(p.second));
}

template <typename T> std::string toString(const std::vector<T>& vec) {
    return tryCompressStr("[\n" + addIndent(toString(vec.begin(), vec.end())) + "]");
}

template <typename K, typename V> std::string toString(const std::map<K, V>& m) {
    return tryCompressStr("[\n" + addIndent(toString(m.begin(), m.end())) + "]");
}

template <typename T> std::string toString(const std::list<T>& l) {
    return tryCompressStr("[\n" + addIndent(toString(l.begin(), l.end())) + "]");
}

std::string serialize(const auto& val) {
    if constexpr (requires { std::string(val); }) {
        return "\"" + std::string(val) + "\"";
    } else if constexpr (requires { toString(val); }) {
        return toString(val);
    } else if constexpr (requires { val.toString(); }) {
        return val.toString();
    } else if constexpr (requires { std::to_string(val); }) {
        return std::to_string(val);
    } else if constexpr (requires { std::ostringstream() << val; }) {
        static std::ostringstream oss;
        oss.str(""), oss.clear(), oss << val;
        return oss.str();
    } else {
        static_assert(false, "can not convert to string");
    }
}

template <typename T> std::string serialize(const std::unique_ptr<T>& ptr) {
    if constexpr (requires { ptr->toString(); }) {
        return ptr ? ptr->toString() : "nullptr";
    } else {
        return serialize(*ptr);
    }
}

template <typename T>
    requires(
        requires(T t) { t.toString(); } || requires(T t) { toString(t); })
struct std::formatter<T> : std::formatter<std::string> {
    auto format(const T& t, std::format_context& ctx) const {
        std::string str;
        if constexpr (requires { t.toString(); })
            str = t.toString();
        else
            str = toString(t);
        return std::formatter<std::string>::format(str, ctx);
    }
};

template <typename T>
struct std::formatter<std::unique_ptr<T>> : std::formatter<std::string> {
    auto format(const std::unique_ptr<T>& ptr, std::format_context& ctx) const {
        return std::formatter<std::string>::format(serialize(ptr), ctx);
    }
};

std::string serializeVar(const char* names, const auto& var, const auto&... rest) {
    std::ostringstream oss;
    const char* comma = strchr(names, ',');
    while (names[0] == ' ') names++;
    if (comma != nullptr) {
        oss.write(names, comma - names) << ": " << serialize(var) << ","
                                        << "\n";
        if constexpr (sizeof...(rest)) oss << serializeVar(comma + 1, rest...);
    } else {
        oss.write(names, strlen(names)) << ": " << serialize(var) << "\n";
    }
    return oss.str();
}

#define serializeClass(name, ...) \
    tryCompressStr(name " {\n" + addIndent(serializeVar(#__VA_ARGS__, __VA_ARGS__)) + "}")

inline std::string
getLocation(std::source_location location = std::source_location::current()) {
    return std::format("{}:{} `{}`",
                       std::filesystem::path(location.file_name()).filename().string(),
                       location.line(), location.function_name());
}

#define RED      "\033[0;31m"
#define L_RED    "\033[1;31m"
#define GREEN    "\033[0;32m"
#define L_GREEN  "\033[1;32m"
#define YELLOW   "\033[0;33m"
#define L_YELLOW "\033[1;33m"
#define BLUE     "\033[0;34m"
#define L_BLUE   "\033[1;34m"
#define PURPLE   "\033[0;35m"
#define L_PURPLE "\033[1;35m"
#define CYAN     "\033[0;36m"
#define L_CYAN   "\033[1;36m"
#define DARK     "\033[2m"
#define RESET    "\033[0m"

#define addLocation(...) \
    "{}{}:{}\n{}", DARK, getLocation(), RESET, addIndent(std::format(__VA_ARGS__), 2)

#ifndef DEBUG
#    define runtimeError(...) \
        throw std::runtime_error(std::format(addLocation(__VA_ARGS__)))
#    define compileError(...) \
        throw std::logic_error(std::format(addLocation(__VA_ARGS__)))
#    define debugLog(...) (void)0
#else
#    define runtimeError(...)                               \
        std::cerr << RED "[runtime error]\n" RESET          \
                  << std::format(addLocation(__VA_ARGS__)), \
            assert(false)
#    define compileError(...)                               \
        std::cerr << RED "[compile error]\n" RESET          \
                  << std::format(addLocation(__VA_ARGS__)), \
            assert(false)
#    define debugLog(...) \
        std::cerr << tryCompressStr(std::format(addLocation(__VA_ARGS__)))
#endif

template <typename... Ts> struct Visitor : Ts... {
    using Ts::operator()...;
};

template <typename... Ts> Visitor(Ts...) -> Visitor<Ts...>;

template <typename T> struct Match {
    T value;
    Match(T&& value) : value(std::forward<T>(value)) {}
    template <typename... Ts> auto operator()(Ts&&... params) {
        return std::visit(Visitor{std::forward<Ts>(params)...}, std::forward<T>(value));
    }
    template <typename... Ts> auto operator()(Visitor<Ts...> visitor) {
        return std::visit(visitor, std::forward<T>(value));
    }
};

template <typename T>
Match(T&&) -> Match<T&&>;  // make for compiler to reserve reference mark

inline std::string
locate(const std::source_location location = std::source_location::current()) {
    std::ostringstream oss;
    oss << std::filesystem::path(location.file_name()).filename().string() << '('
        << location.line() << ':' << location.column() << ") `"
        << location.function_name() << "`";
    return oss.str();
}
#define debug_var(...) \
    std::clog << std::format("{}: {}\n", locate(), show_var(#__VA_ARGS__, __VA_ARGS__))
#define debug(...) \
    std::clog << std::format("{}: {}\n", locate(), std::format(__VA_ARGS__))
std::string show_var(const char* names, const auto& var, const auto&... rest) {
    std::ostringstream oss;
    const char* comma = strchr(names, ',');
    if (comma != nullptr) {
        oss.write(names, comma - names) << " = " << var << ",";
        if constexpr (sizeof...(rest)) oss << show_var(comma + 1, rest...);
    } else {
        oss.write(names, strlen(names)) << " = " << var;
    }
    return oss.str();
}
