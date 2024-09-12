#include "autodiff.h"

#include <format>
#include <iostream>

using namespace std;

int main()
{
    var x1 = 2, x2 = 3;
    // var y = pow(x1, x2);
    var y = x1 ^ x2;
    y.propagate();
    // var y = log(x1) + x1 * x2 - sin(x2);
    // y.propagate();
    cout << std::format("y = {}, py/px1 = {}, py/px2 = {}", y(), x1.diff(), x2.diff()) << endl;
    return 0;
}