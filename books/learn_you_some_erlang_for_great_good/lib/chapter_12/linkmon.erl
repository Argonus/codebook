-module(linkmon).
-author("piotr").

%% API
-compile(export_all).

chain(0) ->
  receive
    X -> X
  after 2000 ->
    exit(normal)
  end;
chain(N) when N > 0 ->
  process_flag(trap_exit, true),
  spawn_link(fun() -> chain(N - 1) end),
  receive
    X -> X
  end.

monitor_chain(N) ->
  spawn_monitor(fun() -> chain(N) end).

start_critic() ->
  spawn(?MODULE, restarter, []).

judge(Band, Album) ->
  Ref = make_ref(),
  critic ! {self(), Ref, {Band, Album}},
  receive
    {Ref, Criticism} -> Criticism
  after 2000 ->
    timeout
  end.

critic() ->
  receive
    {From, Ref, {"Rage Against", "Unify"}} -> {From ! {Ref, "They are great"}};
    {From, Ref, {"System Down", "Memoize"}} -> {From ! {Ref, "They are not great"}}
  end,
  critic().

restarter() ->
  process_flag(trap_exit, true),
  Pid = spawn_link(?MODULE, critic, []),
  register(critic, Pid),
  receive
    {'EXIT', Pid, normal} -> ok;
    {'EXIT', Pid, shutdown} -> ok;
    {'EXIT', Pid, _} -> restarter()
  end.

murder_critic(Pid) ->
  exit(Pid, headshot).