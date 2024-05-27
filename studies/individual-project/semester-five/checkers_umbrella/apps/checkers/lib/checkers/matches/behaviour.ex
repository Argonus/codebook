defmodule Checkers.Matches.Behaviour do
  @moduledoc """
  This module defines the behaviour for the Matches context.
  """
  @type user_id :: non_neg_integer()
  @type match_id :: String.t()
  @type match :: Checkers.Matches.MatchStruct.t()
  @type color :: :black | :white

  @doc """
  Returns match if exists
  """
  @callback get_match(match_id) :: {:ok, match} | {:error, :not_found}

  @doc """
  Creates a match for given host.
  Match is created with basic attributes
  """
  @callback create_match(user_id) :: {:ok, match} | {:error, Ecto.Changeset.t()}

  @doc """
  Assigns another player to match
  """
  @callback join_match(match_id, user_id) :: {:ok, match} | {:error, :not_found | Ecto.Changeset.t()}

  @doc """
  Assigns color for players in match.
  Assigned will be only host color, as player color will be always opposite
  """
  @callback assign_color(match_id, user_id, color) ::
              {:ok, match} | {:error, :not_found | :forbbiden | Ecto.Changeset.t()}

  @doc """
  Deletes existing match if user_id is same as host_id
  """
  @callback delete_match(match_id, user_id) :: :ok | {:error, :not_found | :forbbiden}
end
