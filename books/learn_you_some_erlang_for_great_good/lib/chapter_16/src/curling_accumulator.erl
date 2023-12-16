-module(curling_accumulator).
-behavior(gen_event).
-author("piotr").

-record(state, {teams=orddict:new(), round=0}).

%% API
-export([init/1, handle_event/2, handle_info/2, handle_call/2, code_change/3, terminate/2]).

init([]) ->
  {ok, #state{}}.

handle_event({set_teams, TeamA, TeamB}, S=#state{teams=T}) ->
  Teams = add_teams([TeamA, TeamB], T),
  {ok, S#state{teams=Teams}};
handle_event({add_points, Team, N}, S=#state{teams=T}) ->
  Teams = orddict:update_counter(Team, N, T),
  {ok, S#state{teams=Teams}};
handle_event(next_round, S=#state{round=R}) ->
  {ok, S#state{round=R+1}};
handle_event(_, State) ->
  {ok, State}.

handle_call(game_data, S=#state{teams=T, round=R}) ->
  {ok, {orddict:to_list(T), {round, R}}, S}.

handle_info(_, State) ->
  {ok, State}.

code_change(_OldVsn, State, _Extra) ->
  {ok, State}.

terminate(_Reason, _State) ->
  ok.

add_teams([], Teams) ->
    Teams;
add_teams([H|T], Teams) ->
  add_teams(T, add_team(H, Teams)).
add_team(Team, Teams) ->
  orddict:store(Team, 0, Teams).