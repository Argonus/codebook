defmodule Checkers.Seasons.Behaviour do
  @moduledoc """
  This module defines the behaviour for the Seasons context.
  """
  alias Checkers.Seasons.SeasonStruct

  @doc """
  Starts new season
  """
  @callback init_season() :: {:ok, SeasonStruct.t()} | {:error, Ecto.Changeset.t()}

  @doc """
  Returns current season
  """
  @callback get_current_season() :: {:ok, SeasonStruct.t()} | {:error, :not_found}

  @doc """
  Joins a user to a season
  """
  @callback join_season(season_id :: integer(), user_id :: integer()) ::
              :ok | {:error, :season_not_found | :user_not_found}
end
