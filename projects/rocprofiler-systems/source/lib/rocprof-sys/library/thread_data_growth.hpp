// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/concepts.hpp"
#include "core/containers/stable_vector.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace rocprofsys
{
using grow_functor_t = std::int64_t (*)(std::int64_t);

namespace detail
{
inline auto&
grow_functors()
{
    static auto _v = container::stable_vector<grow_functor_t>{};
    return _v;
}

inline std::mutex&
grow_functors_mutex()
{
    static std::mutex _v;
    return _v;
}
}  // namespace detail

/// Registers a callback that grows one thread_data container. Thread-safe: distinct
/// thread_data specializations register from their own function-local static
/// initializers, which can run concurrently on different threads.
inline void
register_grow_functor(grow_functor_t _func)
{
    auto _lk = std::lock_guard<std::mutex>{ detail::grow_functors_mutex() };
    detail::grow_functors().emplace_back(_func);
}

/// Returns a copy of the registered callbacks. Callers invoke them without the registry
/// lock held, since a callback may initialize a thread_data and register another one.
inline std::vector<grow_functor_t>
get_grow_functors()
{
    auto _lk = std::lock_guard<std::mutex>{ detail::grow_functors_mutex() };
    auto _v  = std::vector<grow_functor_t>{};
    for(auto itr : detail::grow_functors())
        _v.push_back(itr);
    return _v;
}

inline auto&
get_peak_num_threads_callback()
{
    static std::function<std::int64_t()> _v = []() -> std::int64_t {
        return static_cast<std::int64_t>(max_supported_threads);
    };
    return _v;
}

inline std::int64_t
get_current_peak_num_threads()
{
    return get_peak_num_threads_callback()();
}

inline void
set_peak_num_threads_callback(std::function<std::int64_t()> _cb)
{
    get_peak_num_threads_callback() = std::move(_cb);
}

}  // namespace rocprofsys
