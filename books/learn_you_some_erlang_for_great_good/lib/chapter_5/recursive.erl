-module(recursive).
-author("piotr").

%% API
-export([tail_zip/2]).

tail_reverse(L) -> tail_reverse(L,[]).

tail_reverse([],Acc) -> Acc;
tail_reverse([H|T],Acc) -> tail_reverse(T, [H|Acc]).

tail_zip(X,Y) -> tail_reverse(tail_zip(X,Y,[])).

tail_zip([],[],Acc) -> Acc;
tail_zip([X|Xt], [Y|Yt], Acc) -> tail_zip(Xt, Yt, [{X, Y} | Acc]).
