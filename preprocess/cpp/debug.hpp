#pragma once

#include <cstdio>

#ifdef DEBUG
#define GF_PREPROCESS_DEBUG_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define GF_PREPROCESS_DEBUG_LOG(...) ((void)0)
#endif
