-module(ppool).
-behavior(application).
-author("piotr").

%% Callbacks
-export([start/2, stop/1]).
%% API
-export([start_pool/3, stop_pool/1, run/2, sync_queue/2, async_queue/2]).

start(normal, _Args) ->
  ppool_supersup:start_link().

stop(_State) ->
  ppool_supersup:stop().

start_pool(Name, Limit, {M,F,A}) ->
  ppool_supersup:start_pool(Name, Limit, {M,F,A}).

stop_pool(Name) ->
  ppool_supersup:stop_pool(Name).

run(Name, Args) ->
  ppool_serv:run(Name, Args).

async_queue(Name, Args) ->
  ppool_serv:async_queue(Name, Args).

sync_queue(Name, Args) ->
  ppool_serv:sync_queue(Name, Args).
