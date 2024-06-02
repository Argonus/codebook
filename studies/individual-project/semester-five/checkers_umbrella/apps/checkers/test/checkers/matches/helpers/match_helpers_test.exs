defmodule Checkers.Matches.Helpers.MatchHelpersTest do
  use ExUnit.Case, async: true

  alias Checkers.Matches.Helpers.MatchHelpers

  describe "opposite_color/1" do
    test "returns nil for nil" do
      assert nil == MatchHelpers.opposite_color(nil)
    end

    test "returns white for black" do
      assert :white == MatchHelpers.opposite_color(:black)
    end

    test "returns black for white" do
      assert :black == MatchHelpers.opposite_color(:white)
    end

    test "raises error when color is invalid" do
      assert_raise(FunctionClauseError, fn ->
        MatchHelpers.opposite_color(:blue)
      end)
    end
  end

  describe "draw_initial_board/1" do
    test "draw initial board with pawns" do
      assert [
               [
                 %{id: "0x0", color: :black, pawn: :white},
                 %{id: "0x1", color: :white, pawn: nil},
                 %{id: "0x2", color: :black, pawn: :white},
                 %{id: "0x3", color: :white, pawn: nil},
                 %{id: "0x4", color: :black, pawn: :white},
                 %{id: "0x5", color: :white, pawn: nil},
                 %{id: "0x6", color: :black, pawn: :white},
                 %{id: "0x7", color: :white, pawn: nil}
               ],
               [
                 %{id: "1x0", color: :white, pawn: nil},
                 %{id: "1x1", color: :black, pawn: :white},
                 %{id: "1x2", color: :white, pawn: nil},
                 %{id: "1x3", color: :black, pawn: :white},
                 %{id: "1x4", color: :white, pawn: nil},
                 %{id: "1x5", color: :black, pawn: :white},
                 %{id: "1x6", color: :white, pawn: nil},
                 %{id: "1x7", color: :black, pawn: :white}
               ],
               [
                 %{id: "2x0", color: :black, pawn: :white},
                 %{id: "2x1", color: :white, pawn: nil},
                 %{id: "2x2", color: :black, pawn: :white},
                 %{id: "2x3", color: :white, pawn: nil},
                 %{id: "2x4", color: :black, pawn: :white},
                 %{id: "2x5", color: :white, pawn: nil},
                 %{id: "2x6", color: :black, pawn: :white},
                 %{id: "2x7", color: :white, pawn: nil}
               ],
               [
                 %{id: "3x0", color: :white, pawn: nil},
                 %{id: "3x1", color: :black, pawn: nil},
                 %{id: "3x2", color: :white, pawn: nil},
                 %{id: "3x3", color: :black, pawn: nil},
                 %{id: "3x4", color: :white, pawn: nil},
                 %{id: "3x5", color: :black, pawn: nil},
                 %{id: "3x6", color: :white, pawn: nil},
                 %{id: "3x7", color: :black, pawn: nil}
               ],
               [
                 %{id: "4x0", color: :black, pawn: nil},
                 %{id: "4x1", color: :white, pawn: nil},
                 %{id: "4x2", color: :black, pawn: nil},
                 %{id: "4x3", color: :white, pawn: nil},
                 %{id: "4x4", color: :black, pawn: nil},
                 %{id: "4x5", color: :white, pawn: nil},
                 %{id: "4x6", color: :black, pawn: nil},
                 %{id: "4x7", color: :white, pawn: nil}
               ],
               [
                 %{id: "5x0", color: :white, pawn: nil},
                 %{id: "5x1", color: :black, pawn: :black},
                 %{id: "5x2", color: :white, pawn: nil},
                 %{id: "5x3", color: :black, pawn: :black},
                 %{id: "5x4", color: :white, pawn: nil},
                 %{id: "5x5", color: :black, pawn: :black},
                 %{id: "5x6", color: :white, pawn: nil},
                 %{id: "5x7", color: :black, pawn: :black}
               ],
               [
                 %{id: "6x0", color: :black, pawn: :black},
                 %{id: "6x1", color: :white, pawn: nil},
                 %{id: "6x2", color: :black, pawn: :black},
                 %{id: "6x3", color: :white, pawn: nil},
                 %{id: "6x4", color: :black, pawn: :black},
                 %{id: "6x5", color: :white, pawn: nil},
                 %{id: "6x6", color: :black, pawn: :black},
                 %{id: "6x7", color: :white, pawn: nil}
               ],
               [
                 %{id: "7x0", color: :white, pawn: nil},
                 %{id: "7x1", color: :black, pawn: :black},
                 %{id: "7x2", color: :white, pawn: nil},
                 %{id: "7x3", color: :black, pawn: :black},
                 %{id: "7x4", color: :white, pawn: nil},
                 %{id: "7x5", color: :black, pawn: :black},
                 %{id: "7x6", color: :white, pawn: nil},
                 %{id: "7x7", color: :black, pawn: :black}
               ]
             ] ==
               MatchHelpers.draw_initial_board()
    end
  end
end
