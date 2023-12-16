-module(my_server).
-author("piotr").

%% API
-export([call/2, cast/2, replay/2, start/2, start_link/2]).

%% Public API
start(Module, InitState) ->
  spawn(fun() -> init(Module, InitState) end).

start_link(Module, InitState) ->
  spawn_link(fun() -> init(Module, InitState) end).

call(Pid, Msg) ->
  io:format("~p~n", [Msg]),
  Ref = erlang:monitor(process, Pid),
  Pid ! {sync, self(), Ref, Msg},
  receive
    {Ref, Replay} ->
      erlang:demonitor(Ref, [flush]),
      Replay;
    {'DOWN', Ref, process, Pid, Reason} ->
      erlang:error(Reason)
  after 5000 ->
    erlang:error(timeout)
  end.

cast(Pid, Msg) ->
  Pid ! {async, Msg},
  ok.

replay({Pid, Ref}, Replay) ->
  Pid ! {Ref, Replay}.

%% Private Functions
init(Module, InitState) ->
  loop(Module, Module:init(InitState)).

loop(Module, State) ->
  receive
    {async, Message} ->
      loop(Module, Module:handle_cast(Message, State));
    {sync, Pid, Ref, Message} ->
      loop(Module, Module:handle_call(Message, {Pid, Ref}, State))
  end.