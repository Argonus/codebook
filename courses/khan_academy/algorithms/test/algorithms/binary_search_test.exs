defmodule Algorithms.BinarySearchTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Algorithms.BinarySearch

  describe "search/2" do
    test "returns index of a member value" do
      check all(
              list <- uniq_list_of(integer(), min_length: 1),
              member <- member_of(list),
              max_runs: 10
            ) do
        list = Enum.sort(list)

        assert BinarySearch.search(list, member) == Enum.find_index(list, &(&1 == member))
      end
    end

    test "returns -1 when member not part of set" do
      assert BinarySearch.search([1, 2, 3], -1) == -1
    end

    test "returns -1 when input is empty" do
      assert BinarySearch.search([], 1) == -1
    end

    test "returns 0 when list contains only target" do
      assert BinarySearch.search([1], 1) == 0
    end
  end
end
