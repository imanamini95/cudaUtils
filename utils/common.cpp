#include "common.h"
#include <cstdio>

void compareArrays(int *a, int *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Arrays are different \n");
            return;
        }
    }
    printf("Arrays are same \n");
}
