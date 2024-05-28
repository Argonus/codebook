defmodule Checkers.Seasons.Impl do
  @moduledoc """
  Implementation of seasons module
  """
  @behaviour Checkers.Seasons.Behaviour

  alias Checkers.Seasons.SeasonManagement

  @impl true
  defdelegate init_season, to: SeasonManagement

  @impl true
  defdelegate get_current_season, to: SeasonManagement

  @impl true
  defdelegate join_season(season_id, user_id), to: SeasonManagement
end
