defmodule GrokkingAlgorithms.Chapter4.QuickSortTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias GrokkingAlgorithms.Chapter4.QuickSort

  describe "sort/1" do
    test "returns zero when list is empty" do
      assert [] == QuickSort.sort([])
    end

    test "returns element when list has one element" do
      assert [1] == QuickSort.sort([1])
    end

    test "returns length of list" do
      check all(
              list <- list_of(integer(), min_length: 2),
              max_runs: 10
            ) do
        assert QuickSort.sort(list) == Enum.sort(list)
      end
    end
  end
end
