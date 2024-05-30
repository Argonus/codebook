option solver minos;

# Nominal Times
param tA := 4; 
param tB := 8; 
param tC := 8;
param tD := 3;
param tE := 6;
param tF := 10;
param tG := 7;
param tH := 5;
param tI := 4;
param tJ := 5;

var xA >= 0;
var xB >= 0;
var xC >= 0;
var xD >= 0;
var xE >= 0;
var xF >= 0;
var xG >= 0;
var xH >= 0;
var xI >= 0;
var xJ >= 0;
var T >= 0;

# Part One
minimize Czas: T;
# A related
subject to req_xB: xB >= xA + tA;
subject to req_xF: xF >= xA + tA;
subject to req_xG: xG >= xA + tA;
# F related
subject to req_xH: xH >= xF + tF;
# B related
subject to req_xC1: xC >= xB + tB;
# H related
subject to req_xC2: xC >= xH + tH;
subject to req_xI1: xI >= xH + tH;
# G related
subject to req_xI2: xI >=  xG + tG;
# I related
subject to req_xE: xE >= xI + tI;
subject to req_xJ: xJ >= xI + tI;
# C related
subject to req_xC: xD >= xC + tC;
# E related
subject to req_xD: xD >= xE + tE;
# End state
subject to req_T1: T >= xD + tD;
subject to req_T2: T >= xJ + tJ;
