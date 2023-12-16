defmodule GrokkingAlgorithms.Chapter4.LengthTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias GrokkingAlgorithms.Chapter4.Length

  describe "length/1" do
    test "returns zero when list is empty" do
      assert 0 == Length.length([])
    end

    test "returns element when list has one element" do
      assert 1 == Length.length([1])
    end

    test "returns length of list" do
      check all(
              list <- list_of(integer(), min_length: 2),
              max_runs: 10
            ) do
        assert Length.length(list) == length(list)
      end
    end
  end
end
