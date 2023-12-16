-module(rpn_calc).
-author("piotr").

%% API
-export([calculate/1, calculate_test/0]).

calculate(L) when is_list(L) ->
  Stack = string:tokens(L, " "),
  [Res] = rpn(Stack),
  Res.


rpn(Stack) ->
  fold(fun rpn/2, [], Stack).

rpn("+", [N1, N2|Stack]) -> [N2 + N1|Stack];
rpn("-", [N1, N2|Stack]) -> [N2 - N1|Stack];
rpn(X, Stack) -> [to_integer(X)|Stack].

to_integer(X) ->
  case string:to_float(X) of
    {error, no_float} -> list_to_integer(X);
    {F, _} -> F
  end.

fold(_, Res, []) ->
  Res;
fold(F, Res, [H|T]) ->
  fold(F, F(H, Res), T).


calculate_test() ->
  5 = calculate("2 3 +"),
  87 = calculate("90 3 -"),
  ok.