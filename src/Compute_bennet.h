#include "Compute.h"
#include <string>
#include <sstream>
#include <iostream>
#include <complex>
#include <cufft.h>
#include <cufftXt.h>
#include <algorithm>
#include <iterator>

__global__ void d_scalePositions(float* , float* , int , int , float , int);

#ifndef _COMPUTE_AVG_SURFTEN_H_
#define _COMPUTE_AVG_SURFTEN_H_

class Bennet : public Compute {
private:
    int number_stored;
protected:
    float areaChange;
    int   normD;
public:

    void allocStorage() override;
    void doCompute(void);
    void writeResults();
    Bennet(std::istringstream&);
    ~Bennet();

};

#endif