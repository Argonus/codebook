defmodule Checkers.Matches do
  @moduledoc """
  This module is a interface to the Matches context.
  """
  @behaviour Checkers.Matches.Behaviour

  @impl true
  def get_match(match_id), do: impl().get_match(match_id)

  @impl true
  def create_match(host_id), do: impl().create_match(host_id)

  @impl true
  def join_match(match_id, player_id), do: impl().join_match(match_id, player_id)

  @impl true
  def assign_color(match_id, user_id, color), do: impl().assign_color(match_id, user_id, color)

  @impl true
  def delete_match(match_id, player_id), do: impl().delete_match(match_id, player_id)

  defp impl, do: Application.get_env(:checkers, :matches_impl, Checkers.Matches.Impl)
end
