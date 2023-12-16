-module(trade_fsm).
-behavior(gen_fsm).
-author("piotr").
-record(state, {
  name="",
  other,
  ownitems=[],
  otheritems=[],
  monitor,
  from}).

%% Public API
-export([start/1, start_link/1, trade/2, accept_trade/1, make_offer/2, retract_offer/2, ready/1, cancel/1]).
%% gen_fsm callbacks
-export([init/1, handle_event/3, handle_sync_event/4, handle_info/3, terminate/3, code_change/4]).
%% Customer states
-export([idle/2, idle/3, idle_wait/2, idle_wait/3, negotiate/2, negotiate/3, wait/2, ready/2, ready/3]).

%% gen_fsm callbacks
-export([]).

%% Public API
start(Name) ->
  gen_fsm:start(?MODULE, [Name], []).

start_link(Name) ->
  gen_fsm:start_link(?MODULE, [Name], []).

%% Asks for begin session. Returns when/if the other accepts. (Sync, wait for response. Lock fsm)
trade(OwnPid, OtherPid) ->
  gen_fsm:sync_send_event(OwnPid, {negotiate, OtherPid}, 30000).

%% Accept someone's other trade. (Sync, wait for response)
accept_trade(OwnPid) ->
  gen_fsm:sync_send_event(OwnPid, accept_negotiate).

%% Send item to table, to be traded. (Async)
make_offer(OwnPid, Item) ->
  gen_fsm:send_event(OwnPid, {make_offer, Item}).

%% Cancel trade offer (Async)
retract_offer(OwnPid, Item) ->
  gen_fsm:send_event(OwnPid, {retract_offer, Item}).

%% Mention you are ready to trade (Sync)
ready(OwnPid) ->
  gen_fsm:sync_send_event(OwnPid, ready, infinity).

% Cancel transaction. (Global, Sync)
cancel(OwnPid) ->
  gen_fsm:sync_send_all_state_event(OwnPid, cancel).

%% FSM-FSM Communication

%% Ask the other FSM Pid for trade session (Async)
ask_negotiate(OtherPid, OwnPid) ->
  gen_fsm:send_event(OtherPid, {ask_negotiate, OwnPid}).

%% Forward the client message accepting the negotiate (Async)
accept_negotiate(OtherPid, OwnPid) ->
  gen_fsm:send_event(OtherPid, {accept_negoriate, OwnPid}).

%% Forward a client's offer (Async)
do_offer(OtherPid, Item) ->
  gen_fsm:send_event(OtherPid, {do_offer, Item}).


%% Forward a client offers cancellation. (Async)
undo_offer(OtherPid, Item) ->
  gen_fsm:send_event(OtherPid, {undo_offer, Item}).

%% Ask the other side if he is ready to trade (Async)
are_you_ready(OtherPid) ->
  gen_fsm:send_event(OtherPid, are_you_ready).

%% Tells the other FSM that you are not ready yet. (Async)
not_yet(OtherPid) ->
  gen_fsm:send_event(OtherPid, not_yet).

%% Tells the other FSM that user is ready. (Async)
am_rady(OtherPid) ->
  gen_fsm:send_event(OtherPid, 'ready!').

%% Acknowledge that fsm is in ready state (Async)
ack_trade(OtherPid) ->
  gen_fsmm:send_event(OtherPid, ack).

%% Ask if is ready to commit. (Async)
ask_commit(OtherPid) ->
  gen_fsm:send_event(OtherPid, ask_commit).

%% Begin sync commit. Sync (locks)
do_commit(OtherPid) ->
  gen_fsm:sync_send_event(OtherPid, do_commit).

%% Cancel transaction. (Sync, Global, locks)
notify_cancel(OtherPid) ->
  gen_fsm:send_all_state_event(OtherPid, cancel).

%% GenFSM callbacks.
init(Name) ->
  %% {_, starting state, state}
  {ok, idle, #state{name=Name}}.

% Async ask for negotiate. (Other player asks)
idle({ask_negotiate, OtherPid}, S=#state{}) ->
  Ref = monitor(process, OtherPid),
  notice(S, "~[ asked for trade negotiation", [OtherPid]),
  {next_state, idle_wait, S#state{other=OtherPid, monitor=Ref}};
idle(Event, Data) ->
  unexpected(Event, idle),
  {next_state, idle, Data}.

% Sync ask as we are asking
idle({ask_negotiate, OtherPid}, From, S=#state{}) ->
  ask_negotiate(OtherPid, self()), %% Asking
  notice(S, "asking user ~p for trade", [OtherPid]),
  Ref = monitor(process, OtherPid),
  {next_state, idle_wait, S#state{other=OtherPid, monitor=Ref, from=From}};
idle(Event, _From, Data) ->
  unexpected(Event, idel),
  {next_state, idle, Data}.

%% Async as its other player asked
idle_wait({ask_negotiate, OtherPid}, S=#state{other=OtherPid}) ->
  gen_fsm:reply(S#state.from, ok),
  notice(S, "starting negotiation", []),
  {next_state, negotiate, S};
idle_wait({accept_negotiate, OtherPid}, S=#state{other=OtherPid}) ->
  gen_fsm:reply(S#state.from, ok),
  notice(S, "starting negotiation", []),
  {next_state, negotiate, S};
idle_wait(Event, Data) ->
  unexpected(Event, idle_wait),
  {next_state, idle_wait, Data}.

%% Sync Own Pid asked. Remove lock of FSM.
idle_wait(accept_negotiate, _From, S=#state{other=OtherPid}) ->
  accept_negotiate(OtherPid, self()),
  notice(S, "accepting negotiation", []),
  {reply, ok, negotiate, S};
idle_wait(Event, _From, Data) ->
  unexpected(Event, idle_wait),
  {next_state, idle_wait, Data}.

%% Async, As we are adding/removing items from list
%% Client -> FSM communication
negotiate({make_offer, Item}, S=#state{ownitems=OwnItems}) ->
  do_offer(S#state.other, Item),
  notice(S, "offering ~p", [Item]),
  {next_state, negotiate, S#state{ownitems=add(Item, OwnItems)}};
negotiate({retract_offer, Item}, S=#state{ownitems=OwnItems}) ->
  undo_offer(S#state.other, Item),
  notice(S, "cancelling offer on ~p", Item),
  {next_state, negotiate, S#state{ownitems=remove(Item, OwnItems)}};
%% FSM -> FSM communication
negotiate({do_offer, Item}, S=#state{otheritems=OtherItems}) ->
  notice(S, "other player offering ~p", [Item]),
  {next_state, negotiate, S#state{otheritems=add(Item, OtherItems)}};
negotiate({undo_offer, Item}, S=#state{otheritems=OtherItems}) ->
  notice(S, "other player cancelling offer ~p", [Item]),
  {next_state, negotiate, S#state{otheritems=remove(Item, OtherItems)}};

% Async as we are just asking.
negotiate(are_you_ready, S=#state{other=OtherPid}) ->
  io:format("Other user readu to trade~n"),
  notice(S, "Other iuser ready to transfer goods: ~n""You get ~p, The other sides gets ~p", [S#state.otheritems, S#state.ownitems]),
  not_yet(OtherPid),
  {next_statem, negotiate, S};
negotiate(Event, Data) ->
  unexpected(Event, negotiate),
  {next_state, negotiate, Data}.

% Sync, to lock FSM
negotiate(ready, From, S=#state{other=OtherPid}) ->
  are_you_ready(OtherPid),
  notice(S, "asking if ready, waiting.", []),
  {next_state, wait, S#state{from=From}};
negotiate(Event, _From, S) ->
  unexpected(Event, negotiate),
  {next_state, negotiate, S}.

%% Async
%% TO handle changed state
wait({do_offer, Item}, S=#state{otheritems=OtherItems}) ->
  gen_fsm:reply(S#state.from, offer_changed),
  notice(S, "other side offering ~p", [Item]),
  {next_state, negotiate, S#state{otheritems=add(Item, OtherItems)}};
wait({undo_offer, Item}, S=#state{otheritems=OtherItems}) ->
  gen_fsm:reply(S#state.from, offer_changed),
  notice(S, "other side cancelling offering ~p", [Item]),
  {next_state, negotiate, S#state{otheritems=remove(Item, OtherItems)}};
%% To handle ready communication
wait(are_you_ready, S=#state{}) ->
  am_rady(S#state.other),
  notice(S, "asked if ready, and I am. Waiting for same reply", []),
  {next_state, wait, S};
wait(not_yet, S=#state{}) ->
  notice(S, "Other not ready yet", []),
  {next_state, wait, S};
wait('ready!', S=#state{}) ->
  am_rady(S#state.other),
  ack_trade(S#state.other),
  gen_fsm:reply(S#state.from, ok),
  notice(S, "other side is ready. Movoing to ready state", []),
  {next_state, ready, S};
wait(Event, Data) ->
  unexpected(Event, wait),
  {next_state, wait, Data}.

%% Async, still locked by sync. With nested sync.
ready(ack, S=#state{}) ->
  case priority(self(), S#state.other) of
    true ->
      try
        notice(S, "asking for commit", []),
        ready_commit = ask_commit(S#state.other),
        notice(S, "ordering commit", []),
        ok = do_commit(S#state.other),
        noitce(S, "commiting...", []),
        commit(S),
        {stop, normal, S}
      catch
        Class:Reason ->
          notice(S, "commit failed", []),
          {stop, {Class, Reason}, S}
      end;
    false ->
      {next_state, ready, S}
  end;
ready(Event, Data) ->
  unexpected(Event, ready),
  {next_state, ready, Data}.

%% Sync, commit data. Waits
ready(ask_commit, _From, S) ->
  notice(S, "replying to ask commit", []),
  {reply, ready_commit, ready, S};
ready(do_commit, _From, S) ->
  notice(S, "commiting...", []),
  commit(S),
  {stop, normla, ok, S};
ready(Event, _From, Data) ->
  unexpected(Event, ready),
  {next_state, ready, Data}.

commit(S=#state{}) ->
  io:format("Transaction completed for ~n").

%% Hadle global async event
handle_event(cancel, _StateName, S=#state{}) ->
  notice(S, "received cancel Event", []),
  {stop, other_cancelled, S};
handle_event(Event, StateName, Data) ->
  unexpected(Event, StateName),
  {next_state, StateName, Data}.

% Handle global sync event
handle_sync_event(cancel, _From, _StateName, S=#state{}) ->
  notify_cancel(S#state.other),
  notice(S, "cancelling trade. Sending cancel event", []),
  {stop, cancelled, ok, S};
handle_sync_event(Event, _From, StateName, Data) ->
  unexpected(Event, StateName),
  {next_state, StateName, Data}.

%% Handle monitor down PID
handle_info({'DOWN', Ref, process, Pid, Reason}, _, S=#state{other=Pid, monitor=Ref}) ->
  notice(S, "other side dead", []),
  {stop, {other_down, Reason}, S};
handle_info(Info, StateName, Data) ->
  unexpected(Info, StateName),
  {next_state, StateName, Data}.

code_change(_OldVsn, StateName, Data, _Extra) ->
  {ok, StateName, Data}.

terminate(normal, ready, S=#state{}) ->
  notice(S, "FSM leaving.", []);
terminate(_Reason, _StateName, _StateData) ->
  ok.

%% Notifications in IO
notice(#state{name=N}, Str, Args) ->
  io:format("~s: "++Str++"~n", [N|Args]).

unexpected(Msg, State) ->
  io:format("~p received unknown event ~p while in state ~p~n", [self(), Msg, State]).

%% Helpers for storing items
add(Item, Items) ->
  [Item | Items].

remove(Item, Items) ->
  Items -- [Item].

%% Helpers to select transaction owner
priority(OwnPid, OtherPid) when OwnPid > OtherPid -> true;
priority(OwnPid, OtherPid) when OwnPid < OtherPid -> false.