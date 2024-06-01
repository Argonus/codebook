defmodule Checkers.Matches.MatchStruct do
  @moduledoc """
  This module is a struct for the Match.
  """
  use TypedStruct
  import Checkers.Matches.Helpers.MatchHelpers

  @type color :: :black | :white | nil
  @type status :: :initialized | :pending | nil
  @type name :: String.t() | nil
  @type pawn :: :black | :white | nil
  @type field :: %{id: String.t(), color: color, pawn: pawn}
  @type board :: list(list(field))

  typedstruct do
    @typedoc "A match"

    field(:id, String.t())
    field(:season_number, non_neg_integer() | nil)
    field(:host_id, non_neg_integer())
    field(:host_name, name)
    field(:host_color, color)
    field(:player_id, non_neg_integer())
    field(:player_name, name)
    field(:player_color, color)
    field(:status, status)
    field(:board, board)
  end

  @spec build_from_schema(Checkers.Schemas.Match) :: __MODULE__.t()
  def build_from_schema(match_schema) do
    %__MODULE__{
      id: match_schema.id,
      season_number: fetch_season_number(match_schema.season),
      host_id: match_schema.host_id,
      host_name: fetch_login(match_schema.host),
      host_color: match_schema.host_color,
      player_id: match_schema.player_id,
      player_name: fetch_login(match_schema.player),
      player_color: opposite_color(match_schema.host_color),
      status: match_schema.status,
      board: build_board(match_schema.moves)
    }
  end

  defp fetch_season_number(%{season_number: number}), do: number
  defp fetch_season_number(_), do: nil

  defp fetch_login(%{login: login}), do: login
  defp fetch_login(_), do: nil

  defp build_board(map) when map_size(map) == 0, do: draw_initial_board()
end
