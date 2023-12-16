defmodule GrokkingAlgorithms.Chapter4.SearchTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias GrokkingAlgorithms.Chapter4.Search

  describe "search/2" do
    test "returns index of a member value" do
      check all(
              list <- uniq_list_of(integer(), min_length: 1),
              member <- member_of(list),
              max_runs: 10
            ) do
        list = Enum.sort(list)

        assert Search.search(list, member) == Enum.find_index(list, &(&1 == member))
      end
    end

    test "returns nil when member not part of set" do
      assert Search.search([1, 2, 3], -1) == nil
    end

    test "returns nil when input is empty" do
      assert Search.search([], 1) == nil
    end

    test "returns 0 when list contains only target" do
      assert Search.search([1], 1) == 0
    end
  end
end
