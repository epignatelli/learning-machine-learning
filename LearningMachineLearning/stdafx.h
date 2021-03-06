// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifdef _WIN32
#include <windows.h>

#include "targetver.h"

#define NOMINMAX // Solving conflicts with methods min and max (std and windows)
#else
#define APIENTRY
#endif

//#define XTENSOR_USE_XSIMD

#undef VOID
#undef small

#include <xtensor\xarray.hpp>
#include <xtensor-blas\xlinalg.hpp>