#include "autodiff.h"

#include <format>
#include <iostream>

using namespace std;

int main()
{
    variable x1 = 0.5, x2 = 3;
    // variable y = pow(x1, x2);
    variable y = asin(x1);
    y.propagate();
    // variable y = log(x1) + x1 * x2 - sin(x2);
    // y.propagate();
    cout << std::format("y = {}, py/px1 = {}, py/px2 = {}", y(), x1.diff(), x2.diff()) << endl;
    return 0;
}