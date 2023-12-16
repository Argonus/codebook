-module(kitty_server).
-author("piotr").

-record(cat, {name, color=green, description}).

%% API
-export([start_link/0, order_cat/4, close_shop/1, return_cat/2]).
-export([init/1, handle_call/3, handle_cast/2]).

%% Client API

start_link() ->
  my_server:start_link(?MODULE, []).

%% Synchronous Call
order_cat(Pid, Name, Color, Description) ->
  my_server:call(Pid, {order, Name, Color, Description}).

close_shop(Pid) ->
  my_server:call(Pid, terminate).

%% Async Call
return_cat(Pid, Cat = #cat{}) ->
  my_server:cast(Pid, {return, Cat}).

%% Server functions
init([]) ->
  [].

handle_call({order, Name, Color, Description}, From, Cats) ->
  if
    Cats =:= [] ->
      my_server:replay(From, make_cat(Name, Color, Description));
    Cats =/= [] ->
      my_server:replay(From, hd(Cats)),
      tl(Cats)
  end;
handle_call(terminate, From, Cats) ->
  my_server:replay(From, ok),
  terminate(Cats).

handle_cast({return, Cat = #cat{}}, Cats) ->
  [Cat|Cats].

%% Private functions
make_cat(Name, Color, Description) ->
  #cat{name=Name, color=Color, description=Description}.

terminate(Cats) ->
  [io:format("~p was set free.~n", [C#cat.name]) || C <- Cats],
  exit(normal).