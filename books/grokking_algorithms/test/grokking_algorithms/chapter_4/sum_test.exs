defmodule GrokkingAlgorithms.Chapter2.SumTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias GrokkingAlgorithms.Chapter4.Sum

  describe "sum/1" do
    test "returns zero when list is empty" do
      assert 0 == Sum.sum([])
    end

    test "returns element when list has one element" do
      assert 1 == Sum.sum([1])
    end

    test "sums values of list" do
      check all(
              list <- list_of(integer(), min_length: 2),
              max_runs: 10
            ) do
        assert Sum.sum(list) == Enum.reduce(list, 0, &(&2 + &1))
      end
    end
  end
end
