#pragma once
typedef uint16_t u16;

#define MAKE_LOOP(idx1, mIdx1, idx2, mIdx2, idx3, mIdx3) \
for (u16 idx1 = 0; idx1 < mIdx1; idx1++) \
    for (u16 idx2 = 0; idx2 < mIdx2; idx2++) \
        for (u16 idx3 = 0; idx3 < mIdx3; idx3++)
