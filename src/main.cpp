#include "autodiff.h"

#include <iostream>

using namespace std;

int main()
{
    variable x = 2;
    variable y = 2 * x * x;
    y.backpropagate();
    cout << y.diff() << " dx = " << x.diff() << endl;
    return 0;
}