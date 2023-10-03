#pragma once

// [mfs] This is not in use
#if 0

#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

namespace rome {

// https://stackoverflow.com/questions/57726401/stdvariant-vs-inheritance-vs-other-ways-performance
//
// This particular implementation of `visit` is copied from the above
// discussion. When the visitor cannot throw an exception, this code provides
// extremely fast visitation since it eliminates expection catching code. That
// said, if exceptions are possible then it is probably preferred to use a
// lambda that has the following structure:
//
// std::visit([](const auto& v) {
//   using type = std::decay_t<decltype(v)>;
//   if constexpr (std::is_same_v<type, A>) { } 
//   else if (std::is_same_v<type, B>) { }
//   else { } }, var);

template <typename... Fs>
struct overload : Fs... {
  using Fs::operator()...;
};

template <typename... Fs>
overload(Fs...) -> overload<Fs...>;

template <size_t N, typename R, typename Variant, typename Visitor>
[[nodiscard]] constexpr R visit(Variant &&var, Visitor &&vis) {
  if constexpr (N == 0) {
    if (N == var.index()) {
      return std::forward<Visitor>(vis)(
          std::get<0>(std::forward<Variant>(var)));
    }
  } else {
    if (var.index() == N) {
      return std::forward<Visitor>(vis)(
          std::get<N>(std::forward<Variant>(var)));
    }
    return visit<N - 1, R>(std::forward<Variant>(var),
                           std::forward<Visitor>(vis));
  }
  while (true)  // Avoids compilers complaints (-Wreturn-type)
    ;
}

template <class... Args, typename Visitor, typename... Visitors>
[[nodiscard]] constexpr decltype(auto) visit(std::variant<Args...> const &var,
                                             Visitor &&vis,
                                             Visitors &&...visitors) {
  auto ol =
      overload{std::forward<Visitor>(vis), std::forward<Visitors>(visitors)...};
  using result_t = decltype(std::invoke(std::move(ol), std::get<0>(var)));

  static_assert(sizeof...(Args) > 0);
  return visit<sizeof...(Args) - 1, result_t>(var, std::move(ol));
}

template <class... Args, typename Visitor, typename... Visitors>
[[nodiscard]] constexpr decltype(auto) visit(std::variant<Args...> &var,
                                             Visitor &&vis,
                                             Visitors &&...visitors) {
  auto ol =
      overload{std::forward<Visitor>(vis), std::forward<Visitors>(visitors)...};
  using result_t = decltype(std::invoke(std::move(ol), std::get<0>(var)));

  static_assert(sizeof...(Args) > 0);
  return visit<sizeof...(Args) - 1, result_t>(var, std::move(ol));
}

template <class... Args, typename Visitor, typename... Visitors>
[[nodiscard]] constexpr decltype(auto) visit(std::variant<Args...> &&var,
                                             Visitor &&vis,
                                             Visitors &&...visitors) {
  auto ol =
      overload{std::forward<Visitor>(vis), std::forward<Visitors>(visitors)...};
  using result_t =
      decltype(std::invoke(std::move(ol), std::move(std::get<0>(var))));

  static_assert(sizeof...(Args) > 0);
  return visit<sizeof...(Args) - 1, result_t>(std::move(var), std::move(ol));
}

template <typename Value, typename... Visitors>
inline constexpr bool is_visitable_v = (std::is_invocable_v<Visitors, Value> or
                                        ...);

}  // namespace rome

#endif