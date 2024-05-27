defmodule Checkers.Matches.Impl do
  @moduledoc false
  @behaviour Checkers.Matches.Behaviour

  alias Checkers.Matches.MatchManagement

  @impl true
  defdelegate get_match(match_id), to: MatchManagement

  @impl true
  defdelegate create_match(host_id), to: MatchManagement

  @impl true
  defdelegate join_match(match_id, player_id), to: MatchManagement

  @impl true
  defdelegate delete_match(match_id, user_id), to: MatchManagement

  @impl true
  defdelegate assign_color(match_id, user_id, color), to: MatchManagement
end
