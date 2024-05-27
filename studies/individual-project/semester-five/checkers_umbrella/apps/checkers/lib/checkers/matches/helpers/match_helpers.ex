defmodule Checkers.Matches.Helpers.MatchHelpers do
  @moduledoc """
  This module contains functions that are shared between different modules
  """

  @doc """
  Returns opposite checker color
  """
  @spec opposite_color(nil) :: nil
  @spec opposite_color(:black) :: :white
  @spec opposite_color(:white) :: :black
  def opposite_color(nil), do: nil
  def opposite_color(:black), do: :white
  def opposite_color(:white), do: :black
end
