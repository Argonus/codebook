% Family tree

% Child Facts
child(jan, adam).
child(jasio, jan).
child(stasio, jan).
child(piotr, jan).
child(krzysio,tomek).
child(julia,tomek).

child(jan, maria).
child(jasio, anna).
child(stasio, anna).
child(piotr, anna).
child(krzysio, anna).
child(julia,anna).

% Sex
woman(maria).
woman(anna).
woman(julia).

% Wife Facts
marriage(maria, adam).
marriage(anna, jan).

% Rules
man(X):- \+ woman(X).

% Lvl 0
natural_sibilings(X,Y):- mother(X,M),mother(Y,M),father(X,F),father(Y,F),X \= Y.
step_siblings(X,Y):- mother(X,M),mother(Y,M),father(X,O1),father(Y,O2),X \= Y,O1 \= O2.
step_siblings(X,Y):- mother(X,O1),mother(Y,O2),father(X,F),father(Y,F),X \= Y,O1 \= O2.
sibilings(X,Y):- natural_sibilings(X,Y) ; step_siblings(X,Y).

sister(X,Y):- sibilings(X,Y),woman(Y).
brother(X,Y):- sibilings(X,Y),man(Y).

wife(X,Y):- man(X),marriage(Y,X),woman(Y).
husband(X,Y):- woman(X),marriage(X,Y),man(Y).

% Lvl 1
parent(X,Y):- child(X, Y).
mother(X,Y):- parent(X,Y),woman(Y).
father(X,Y):- parent(X,Y),man(Y).

% Lvl 2
grandchild(X,Y):- child(D, Y),child(X,D).
grandfather(X,Y):- child(Y, F),child(F,X).
grandmother(X,B):- grandfather(D,X),marriage(B,D),woman(B).

% Lvl N
descendant(X,Y):- child(X,Y).
descendant(X,Y):- child(D,Y),descendant(X,D).

ancestor(X,Y):- parent(X,Y).
ancestor(X,Y):- parent(D,Y),ancestor(X,D).