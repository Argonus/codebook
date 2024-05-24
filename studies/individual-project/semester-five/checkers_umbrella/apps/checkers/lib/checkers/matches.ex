defmodule Checkers.Matches do
  @moduledoc """
  This module is a interface to the Matches context.
  """
  @behaviour Checkers.Matches.Behaviour

  @impl true
  def create_match(host_id), do: impl().create_match(host_id)

  @impl true
  def join_match(match_id, player_id), do: impl().join_match(match_id, player_id)

  defp impl, do: Application.get_env(:checkers, :matches_impl, Checkers.Matches.Impl)
end
