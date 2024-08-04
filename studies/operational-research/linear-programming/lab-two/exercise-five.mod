option solver cplex;

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

# Minimal Times
param pA := 1; 
param pB := 6; 
param pC := 5;
param pD := 1;
param pE := 1;
param pF := 2;
param pG := 3;
param pH := 3;
param pI := 3;
param pJ := 2;

# Costs
param cA := 100;
param cB := 40;
param cC := 80;
param cD := 50;
param cE := 20;
param cF := 20;
param cG := 100;
param cH := 100;
param cI := 10;
param cJ := 40;

# Start times vars
var xA >= 0 integer;
var xB >= 0 integer;
var xC >= 0 integer;
var xD >= 0 integer;
var xE >= 0 integer;
var xF >= 0 integer;
var xG >= 0 integer;
var xH >= 0 integer;
var xI >= 0 integer;
var xJ >= 0 integer;
var T >= 0 integer;

# Reduction vars
param zA = tA - pA integer;
param zB = tB - pB integer;
param zC = tC - pC integer;
param zD = tD - pD integer;
param zE = tE - pE integer; 
param zF = tF - pF integer;
param zG = tG - pG integer;
param zH = tH - pH integer;
param zI = tI - pI integer;
param zJ = tJ - pJ integer;


# Part One
minimize Czas: T;
# A related
subject to req_xB: xB >= xA + tA - zA;
subject to req_xF: xF >= xA + tA - zA;
subject to req_xG: xG >= xA + tA - zA;
# F related
subject to req_xH: xH >= xF + tF - zF;
# B related
subject to req_xC1: xC >= xB + tB - zB;
# H related
subject to req_xC2: xC >= xH + tH - zH;
subject to req_xI1: xI >= xH + tH - zH;
# G related
subject to req_xI2: xI >= xG + tG - zG;
# I related
subject to req_xE: xE >= xI + tI - zI;
subject to req_xJ: xJ >= xI + tI - zI;
# C related
subject to req_xC: xD >= xC + tC - zC;
# E related
subject to req_xD: xD >= xE + tE - zE;
# End state
subject to req_T1: T >= xD + tD - zD;
subject to req_T2: T >= xJ + tJ - zJ;

display zA * cA + zB * cB + zC * cC + zD * cD + zE * cE + zF * cF + zG * cG + zH * cH + zI * cI + zJ * cJ;

