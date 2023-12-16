-module(road).
-author("piotr").

%% API
-export([main/1]).

main(ReadFileName) ->
  {ok, Bin} = file:read_file(ReadFileName),
  {ok, Map} = parse_map(Bin),
  {ok, _Dis, Path} = optimal_path(Map),
  io:format("~p~n", [Path]),
  erlang:halt().

parse_map(Bin) when is_binary(Bin) ->
  Map_as_list = binary_to_list(Bin),
  parse_map(Map_as_list);
parse_map(Str) when is_list(Str) ->
  Points = [list_to_integer(X) || X <- string:tokens(Str, "\r\n\t ")],
  Map = group_vals(Points, []),
  {ok, Map}.

group_vals([], Acc) ->
  lists:reverse(Acc);
group_vals([A, B, X|T], Acc) ->
  group_vals(T, [{A, B, X}|Acc]).

optimal_path(Map) ->
  {A, B} = lists:foldl(fun shortest_step/2, {{0, []}, {0, []}}, Map),
  {Dis, Path} = if hd(element(2, A)) =/= {x, 0} -> A;
                    hd(element(2, B)) =/= {x, 0} -> B
                 end,
  Path2 = lists:reverse(Path),
  {ok, Dis, Path2}.

shortest_step({A, B, X}, {{DistA, PathA}, {DistB, PathB}}) ->
  OptA1 = {DistA + A, [{a,A}|PathA]},
  OptA2 = {DistA + B + X, [{x,X}, {b,B}|PathA]},
  OptB1 = {DistB + B, [{b,B}|PathB]},
  OptB2 = {DistA + A + X, [{x, X}, {a, A}|PathA]},
  {erlang:min(OptA1, OptA2), erlang:min(OptB1, OptB2)}.