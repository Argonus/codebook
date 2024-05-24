defmodule Checkers.Matches.Behaviour do
  @moduledoc """
  This module defines the behaviour for the Matches context.
  """
  @type user_id :: non_neg_integer()
  @type match_id :: String.t()
  @type match :: Checkers.Matches.MatchStruct.t()

  @doc """
  Creates a match for given host.
  Match is created with basic attributes
  """
  @callback create_match(user_id) :: {:ok, match} | {:error, Ecto.Changeset.t()}

  @doc """
  Assigns another player to match
  """
  @callback join_match(match_id, user_id) :: {:ok, match} | {:error, :not_found | Ecto.Changeset.t()}
end
