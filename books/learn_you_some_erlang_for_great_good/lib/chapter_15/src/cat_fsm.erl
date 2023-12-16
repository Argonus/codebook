-module(cat_fsm).
-author("piotr").

%% API
-export([start/0, event/2]).

start() ->
  spawn(fun() -> dont_give_a_crap() end).

event(Pid, Event) ->
  Ref = make_ref(),
  Pid ! {self(), Ref, Event},
  receive
    {Ref, Msg} -> {ok, Msg}
  after 5000 ->
    {error, timeout}
  end.

dont_give_a_crap() ->
  receive
    {Pid, Ref, _Msg} -> Pid ! {Ref, meh}
  end,
  io:format("Switching to 'dont_give_a_crap' state ~n"),
  dont_give_a_crap().
