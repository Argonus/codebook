defmodule Chapter2.Exercise1Test do
  use ExUnit.Case, async: true

  alias Chapter2.Exercise1

  doctest Exercise1

  describe "rec_sum/1" do
    test "returns 0 for empty list" do
      assert 0 == Exercise1.rec_sum([])
    end

    test "returns value for list with one element" do
      assert 1 == Exercise1.rec_sum([1])
    end

    test "returns sum of all elements in list" do
      assert 10 == Exercise1.rec_sum([1, 2, 3, 4])
    end
  end

  describe "tail_rec_sum/1" do
    test "returns 0 for empty list" do
      assert 0 == Exercise1.tail_rec_sum([])
    end

    test "returns value for list with one element" do
      assert 1 == Exercise1.tail_rec_sum([1])
    end

    test "returns sum of all elements in list" do
      assert 10 == Exercise1.tail_rec_sum([1, 2, 3, 4])
    end
  end
end
