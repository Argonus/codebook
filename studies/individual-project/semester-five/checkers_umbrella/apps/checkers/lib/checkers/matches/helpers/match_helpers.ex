defmodule Checkers.Matches.Helpers.MatchHelpers do
  @moduledoc """
  This module contains functions that are shared between different modules
  """
  @type color :: :black | :white | nil
  @type pawn :: :black | :white | nil
  @type field :: %{id: String.t(), color: color, pawn: pawn}

  @doc """
  Returns opposite checker color
  """
  @spec opposite_color(nil) :: nil
  @spec opposite_color(:black) :: :white
  @spec opposite_color(:white) :: :black
  def opposite_color(nil), do: nil
  def opposite_color(:black), do: :white
  def opposite_color(:white), do: :black

  @doc """
  Returns initial board
  """
  @spec draw_initial_board :: list(list(field))
  def draw_initial_board do
    Enum.map(0..7, fn row ->
      Enum.map(0..7, fn col ->
        color = calculate_color(row, col)
        pawn = calculate_init_pawn(color, row, col)

        build_field(row, col, color, pawn)
      end)
    end)
  end

  defp calculate_color(row, col), do: if((row + col) |> rem(2) == 0, do: :black, else: :white)

  defp calculate_init_pawn(:white, _, _), do: nil
  defp calculate_init_pawn(:black, row, col) when row < 3, do: :white
  defp calculate_init_pawn(:black, row, col) when row > 4, do: :black
  defp calculate_init_pawn(:black, _, _), do: nil

  defp build_field(row, col, color, pawn), do: %{id: "#{row}x#{col}", color: color, pawn: pawn}
end
