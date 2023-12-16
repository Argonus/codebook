-module(exceptions).
-author("piotr").

%% API
-export([check_throw/0]).

check_throw() ->
  erlang:display("OK"),
  throw("Wait what?"),
  erlang:display("not exactly").

