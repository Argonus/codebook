-module(event).
-author("piotr").
-record(state, {server, name="", to_g=0}).

%% API
-export([start_link/2, start/2, init/3, cancel/1]).

start(EventName, DateTime) ->
  spawn(?MODULE, init, [self(), EventName, DateTime]).

start_link(EventName, DateTime) ->
  spawn_link(?MODULE, init, [self(), EventName, DateTime]).

init(Server, EventName, DateTime) ->
  loop(#state{server=Server, name=EventName, to_g=time_to_go(DateTime)}).

cancel(Pid) ->
  Ref = erlang:monitor(process, Pid),
  Pid ! {self(), Ref, cancel},
  receive
    {Ref, ok} -> erlang:demonitor(Ref, [flush]),
      ok;
    {'DOWN', Ref, process, Pid, _Reason} ->
      ok
  end.

time_to_go(TimeOut={{_, _, _}, {_, _, _}}) ->
  Now = calendat:local_time(),
  ToGo = calendar:datetime_to_gregorian_seconds(TimeOut) - calendar:datetime_to_gregorian_seconds(Now),
  Sec = if
          ToGo > 0 -> ToGo;
          ToGo =< 0 -> 0
        end,
   normalize(Sec).

loop(S=#state{server=Server, to_g=[T|Next]}) ->
  receive
    {Server, Ref, cancel} -> Server ! {Ref, ok}
  after timer:seconds(T) ->
    if
      Next =:= [] ->
        Server ! {done, S#state.name};
      Next =/= [] ->
        loop(S#state{to_g=Next})
    end
  end.

normalize(N) ->
  Limit = 49*24*60*50,
  [N rem Limit | lists:duplicate(N div Limit, Limit)].