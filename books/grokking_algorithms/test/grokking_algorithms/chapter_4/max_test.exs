defmodule GrokkingAlgorithms.Chapter2.MaxTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias GrokkingAlgorithms.Chapter4.Max

  describe "max/1" do
    test "returns nil when list is empty" do
      assert nil == Max.max([])
    end

    test "returns element when list has one element" do
      assert 1 == Max.max([1])
    end

    test "returns element when list has negative elements" do
      assert -1 == Max.max([-1, -1])
    end

    test "returns max elem of given list of integers" do
      check all(
              list <- list_of(integer(), min_length: 2),
              max_runs: 10
            ) do
        assert Max.max(list) == Enum.max(list)
      end
    end
  end
end
