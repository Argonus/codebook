-module(m8ball_sup).
-behavior(supervisor).
-author("piotr").

%% API
-export([start_link/0, init/1]).

start_link() ->
  supervisor:start_link({global, ?MODULE}, ?MODULE, []).

init([]) ->
  MaxRestart = 1,
  MaxTime = 10,
  {
    ok,
    {
      {one_for_one, MaxRestart, MaxTime},
      [
        {m8ball, {m8ball_server, start_link, []}},
        permanent,
        5000,
        worker,
        [m8ball_server]
      ]
    }
  }.