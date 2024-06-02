defmodule Checkers.Seasons do
  @moduledoc """
  This module is a interface to the Seasons context.
  """
  @behaviour Checkers.Seasons.Behaviour

  @impl true
  def init_season(), do: impl().init_season()

  @impl true
  def get_current_season(), do: impl().get_current_season()

  @impl true
  def join_season(season_id, user_id), do: impl().join_season(season_id, user_id)

  defp impl, do: Application.get_env(:checkers, :seasons_impl, Checkers.Seasons.Impl)
end
