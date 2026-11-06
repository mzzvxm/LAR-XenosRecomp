#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif

#include <dxcapi.h>

#include <bit>
#include <cassert>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <map>
#include <smolv.h>
#include <fmt/core.h>
#include <string>
#include <unordered_map>
#include <xxhash.h>
#include <zstd.h>
