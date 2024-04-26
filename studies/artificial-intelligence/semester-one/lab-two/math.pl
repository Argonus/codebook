factorial(1,1).
factorial(N,F):-
    N > 1,
    N1 is N - 1,
    factorial(N1,F1),
    F is N * F1.

power(X,0,1).
power(X,N,Y):-
    N > 0,
    N1 is N - 1,
    power(X,N1,Y1),
    Y is X * Y1.