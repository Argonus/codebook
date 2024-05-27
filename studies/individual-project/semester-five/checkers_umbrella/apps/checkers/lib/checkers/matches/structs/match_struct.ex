defmodule Checkers.Matches.MatchStruct do
  @moduledoc """
  This module is a struct for the Match.
  """
  use TypedStruct
  import Checkers.Matches.Helpers.MatchHelpers

  @type color :: :black | :white | nil
  @type status :: :initialized | :pending | nil

  typedstruct do
    @typedoc "A match"

    field(:id, String.t())
    field(:host_id, non_neg_integer())
    field(:host_color, color)
    field(:player_id, non_neg_integer())
    field(:player_color, color)
    field(:status, status)
  end

  @spec build_from_schema(Checkers.Schemas.Match) :: __MODULE__.t()
  def build_from_schema(match_schema) do
    %__MODULE__{
      id: match_schema.id,
      host_id: match_schema.host_id,
      host_color: match_schema.host_color,
      player_id: match_schema.player_id,
      player_color: opposite_color(match_schema.host_color),
      status: match_schema.status
    }
  end
end
