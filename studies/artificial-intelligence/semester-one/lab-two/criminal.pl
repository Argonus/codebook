% fakty
osoba(tomasz,55,stolarz).
osoba(krzysztof,25,piłkarz).
osoba(krzysztof,25,rzeźnik).
osoba(piotr,25,złodziej).
osoba(anna,39,chirurg).

romans(anna,piotr).
romans(anna,krzysztof).
romans(agnieszka,piotr).
romans(agnieszka,tomasz).

zamordowana(agnieszka).

prawdopodobnie_zamordowana(agnieszka,kij_golfowy).
prawdopodobnie_zamordowana(agnieszka,łom).
pobrudzony(tomasz,krew).
pobrudzony(agnieszka,krew).
pobrudzony(krzysztof,krew).
pobrudzony(krzysztof,błoto).
pobrudzony(piotr,błoto).
pobrudzony(anna,krew).
posiada(tomasz,sztuczna_noga).
posiada(piotr,rewolwer).

podobne_obrażenia(sztuczna_noga,kij_golfowy).
podobne_obrażenia(noga_od_stołu,kij_golfowy).
podobne_obrażenia(łom,kij_golfowy).
podobne_obrażenia(nożyczki,nóż).
podobne_obrażenia(but_piłkarski,kij_golfowy).
mężczyzna(piotr).
mężczyzna(krzysztof).
mężczyzna(tomasz).
kobieta(anna).
kobieta(agnieszka).

%reguły
posiada(X,but_pilkarski):- osoba(X,_,piłkarz).
posiada(X,piłka):- osoba(X,_,piłkarz).
posiada(X,nóż):- osoba(X,_,rzeźnik).
posiada(X,nóż):- osoba(X,_,chirurg).
posiada(X,nożyczki):- osoba(X,_,chirurg).
posiada(X,łom):- osoba(X,_,złodziej).
posiada(X,noga_od_stołu):- osoba(X,_,stolarz).

podejrzany(X):-
    zamordowana(Z),
    prawdopodobnie_zamordowana(Z,Y),
    podobne_obrażenia(N,Y),
    posiada(X,N).

motyw(X,zazdrosc):-
    mężczyzna(X),
    zamordowana(Z),
    romans(Z,X),
    X \= Z.

motyw(X,zazdrosc):-
    kobieta(X),
    zamordowana(Z),
    romans(Z,Y),
    romans(X,Y),
    mężczyzna(Y),
    X \= Z.

motyw(X,pieniadze):- mężczyzna(X),osoba(X,_,złodziej).


morderca(X):-
    podejrzany(X),
    motyw(X,_),
    zamordowana(Z),
    pobrudzony(Z,S),
    pobrudzony(X,S).

motyw_mordercy(M):- morderca(X),motyw(X,M).