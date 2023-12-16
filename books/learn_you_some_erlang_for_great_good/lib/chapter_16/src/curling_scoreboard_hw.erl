-module(curling_scoreboard_hw).
-author("piotr").

%% API
-export([set_teams/2, add_point/1, next_round/0, reset_board/0]).

set_teams(TeamA, TeamB) ->
  io:format("Scoreboard: Team ~s vs Team ~s~n", [TeamA, TeamB]).

add_point(Team) ->
  io:format("Scoreboard: increased score of team ~s by 1~n", [Team]).

next_round() ->
  io:format("Scoreboard: Round over~n").

reset_board() ->
  io:format("Scoreboard: ALl teams are undefined and all scores are 0~n").


