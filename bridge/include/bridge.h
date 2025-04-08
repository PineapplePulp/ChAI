#ifndef BRIDGE_H
#define BRIDGE_H

#include <stdio.h>

#ifdef __cplusplus
#include <torch/torch.h>
#include <iostream>
#include <vector>
extern "C" {
#endif

int baz(void);

void wrHello(void);

void wrHelloTorch(void);

float sumArray(float* arr, int* sizes, int dim);

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_H
//hello