defmodule Checkers.Matches.MatchStruct do
  @moduledoc """
  This module is a struct for the Match.
  """
  use TypedStruct

  typedstruct do
    @typedoc "A match"

    field(:id, String.t())
    field(:host_id, non_neg_integer())
    field(:player_id, non_neg_integer())
    field(:status, :initialized)
  end

  @spec build_from_schema(Checkers.Schemas.Match) :: __MODULE__.t()
  def build_from_schema(match_schema) do
    %__MODULE__{
      id: match_schema.id,
      host_id: match_schema.host_id,
      player_id: match_schema.player_id,
      status: match_schema.status
    }
  end
end
