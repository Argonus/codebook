option solver minos;

var x >= 0;
var y >= 0;
maximize z: 3*x + 4*y;
subject to ogr1: 2*x + 3*y <= 36;
subject to ogr2: 2*x + 6*y <= 42;
subject to ogr3: 2*x + y <= 24;