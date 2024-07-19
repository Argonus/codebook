option solver minos;

param c1 := 5; 
param c2 := 3; 
param c3 := 4;
param c4 := 6;

var x1 >= 0;
var x2 >= 0;
var x3 >= 0;
var x4 >= 0;

# Part One
# maximize Profit: c1*x1 + c2*x1;
# subject to Machine1: 6*x1 + 1*x2 <= 30;
# subject to Machine2: 7*x1 + 3*x2 <= 75;
# subject to Machine3: 5*x1 + 8*x2 <= 80;


# Part Two
# maximize Profit: c1*x1 + c2*x1 + c3*x3 + c4*x4;

# subject to Machine1: 6*x1 + 1*x2 + 2*x3 + 6*x4 <= 30;
# subject to Machine2: 7*x1 + 3*x2 + 4*x3 + 6*x4 <= 75;
# subject to Machine3: 5*x1 + 8*x2 + 9*x3 + 7*x4 <= 80;

# Part Three
# Funkcja celu
minimize TotalCost: c1*x1 + c2*x2;

# Ograniczenia
subject to M1: 6*x1 + 1*x2 >= 30;
subject to M2: 7*x1 + 3*x2 >= 75;
subject to M3: 5*x1 + 8*x2 >= 80;