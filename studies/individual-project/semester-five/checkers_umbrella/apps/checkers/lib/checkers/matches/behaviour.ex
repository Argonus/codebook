defmodule Checkers.Matches.Behaviour do
  @moduledoc """
  This module defines the behaviour for the Matches context.
  """
  @type host_id :: non_neg_integer()
  @type match :: Checkers.Matches.MatchStruct.t()

  @callback create_match(host_id) :: {:ok, match} | {:error, Ecto.Changeset.t()}
end
