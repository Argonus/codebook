-module(dog_fsm).
-author("piotr").

%% API
-export([start/0, squirrel/1, pet/1, bark/0]).

start() ->
  spawn(fun() -> bark() end).

squirrel(Pid) ->
  Pid ! squirrel.

pet(Pid) ->
  Pid ! pet.

bark() ->
  io:format("Dog says: BARK! BARK! BARK!~n"),
  receive
    pet ->
      wag_tail();
    _ ->
      io:format("Dog is confused. ~n"),
      bark()
  after 2000 ->
    bark()
  end.

wag_tail() ->
  io:format("Dog wags its tail~n"),
  receive
    pet ->
      sit();
    _ ->
      io:format("Dog is confused. ~n"),
      wag_tail()
  after 2000 ->
    bark()
  end.

sit() ->
  io:format("Dog is sitting. Goooood boy! ~n"),
  receive
    squirrel ->
      bark();
    _ ->
      io:format("Dog is confused. ~n"),
      sit()
  end.




