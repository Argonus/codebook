-module(event_server).
-author("piotr").
-record(state, {events, clients}).
-record(event, {name="", description="", pid, timeout={{1970, 1, 1},{0, 0, 0}}}).

%% API
-export([start/0, start_link/0, subscribe/1, add_event/3, cancel/1, listen/1, init/0]).

start() ->
  register(?MODULE, Pid=spawn(?MODULE, init, [])),
  Pid.

start_link() ->
  register(?MODULE, Pid=spawn_link(?MODULE, init, [])),
  Pid.

subscribe(Pid) ->
  Ref = erlang:monitor(process, whereis(?MODULE)),
  ?MODULE ! {self(), Ref, {subscribe, Pid}},
  receive
    {Ref, ok} ->
      {ok, Ref};
    {'DOWN', Ref, process, _Pid, Reason} ->
      {error, Reason}
  after 5000 ->
    {error, timeout}
  end.

add_event(Name, Description, TimeOut) ->
  Ref = make_ref(),
  ?MODULE ! {self(), Ref, {add, Name, Description, TimeOut}},
  receive
    {Ref, {error, Reason}} -> erlang:error(Reason);
    {Ref, Msg} -> Msg
  after 5000 ->
    {error, timeout}
  end.

cancel(Name) ->
  Ref = make_ref(),
  ?MODULE ! {self(), Ref, {cancel, Name}},
  receive
    {Ref, ok} -> ok
  after 5000 ->
    {error, timeout}
  end.


listen(Delay) ->
  receive
    M = {done, _Name, _Description} -> [M, listen(0)]
  after Delay ->
    []
  end.

init() ->
  NewState = #state{events=orddict:new(), clients=orddict:new()},
  loop(NewState).

loop(S=#state{}) ->
  receive
    {Pid, MsgRef, {subscribe, Client}} ->
      Ref = erlang:monitor(process, Client),
      NewClients = orddict:store(Ref, Client, S#state.clients),
      Pid ! {MsgRef, ok},
      loop(S#state{clients = NewClients});

    {Pid, MsgRef, {add, Name, Description, TimeOut}} ->
      case valid_datetime(TimeOut) of
        true ->
          EventPid = event:start_link(Name, TimeOut),
          NewEvent = #event{name = Name, description = Description, pid = EventPid, timeout = TimeOut},
          NewEvents = orddict:store(Name, NewEvent, S#state.events),
          Pid ! {MsgRef, ok},
          loop(S#state{events = NewEvents});
        false ->
          Pid ! {MsgRef, {error, bad_timeout}},
          loop(S)
      end;

    {Pid, MsgRef, {cancel, Name}} ->
      Events = case orddict:find(Name, S#state.events) of
                 {ok, E} ->
                   event:cancel(E#event.pid),
                   orddict:erase(Name, S#state.events);
                 error ->
                   S#state.events
               end,
      Pid ! {MsgRef, ok},
      loop(S#state{events = Events});

    {done, Name} ->
      case orddict:find(Name, S#state.events) of
        {ok, E} ->
          send_to_clients({done, E#event.name, E#event.description}, S#state.clients),
          NewEvents = orddict:erase(Name, S#state.events),
          loop(S#state{events = NewEvents});
        error ->
          loop(S)
      end;

    shutdown ->
      exit(shutdown);

    {'DOWN', Ref, process, _Pid, _Reason} ->
      NewClients = orddict:erase(Ref, S#state.clients),
      loop(S#state{clients = NewClients});

    code_change ->
      ?MODULE:loop(S);

    Unknown ->
      io:format("Unknow message: ~p~n", [Unknown]),
      loop(S)
  end.

send_to_clients(Msg, ClientDict) ->
  orddict:map(fun(_Ref, Pid) -> Pid ! Msg end, ClientDict).

valid_datetime({Date, Time}) ->
  try
    calendar:valid_date(Date) andalso valid_time(Time)
  catch
    error:function_clause -> false
  end;
valid_datetime(_) ->
  false.

valid_time({H, M, S}) -> valid_time(H, M, S).
valid_time(H, M, S) when
  H >= 0, H < 24,
  M >= 0, M < 60,
  S >= 0, S < 60
  -> true;
valid_time(_, _, _)
  -> false.