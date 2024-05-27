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
end
