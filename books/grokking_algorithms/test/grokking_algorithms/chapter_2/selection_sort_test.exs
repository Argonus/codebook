defmodule GrokkingAlgorithms.Chapter2.SelectionSortTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias GrokkingAlgorithms.Chapter2.SelectionSort

  describe "sort/1" do
    test "returns empty list" do
      assert [] == SelectionSort.sort([])
    end

    test "returns list with one element" do
      assert [1] == SelectionSort.sort([1])
    end

    test "sorts given list of integers" do
      check all(
              list <- uniq_list_of(integer(), min_length: 2),
              max_runs: 10
            ) do
        assert SelectionSort.sort(list) == Enum.sort(list)
      end
    end

    test "sorts given list of binaries" do
      check all(
              list <- uniq_list_of(binary(), min_length: 2),
              max_runs: 10
            ) do
        assert SelectionSort.sort(list) == Enum.sort(list)
      end
    end
  end
end
