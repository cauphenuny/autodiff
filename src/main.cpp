#include "autodiff.h"
#include "util.h"

#include <format>
#include <iostream>

using namespace std;

int main()
{
    variable x1 = 2, x2 = 5;
    variable y = log(x1) + x1 * x2 - sin(x2);
    y.backpropagate();
    cout << std::format("y = {}, py/px1 = {}, py/px2 = {}", y(), x1.diff(), x2.diff()) << endl;
    return 0;
}