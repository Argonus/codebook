defmodule GrokkingAlgorithms.Chapter9.BackpackTest do
  use ExUnit.Case, async: true

  alias GrokkingAlgorithms.Chapter9.Backpack

  @guitar %{name: "Guitar", weight: 1, cost: 1500}
  @stereo %{name: "Stereo", weight: 4, cost: 3000}
  @laptop %{name: "Laptop", weight: 3, cost: 3000}
  @iphone %{name: "IPhone", weight: 1, cost: 2000}
  @mp3 %{name: "MP3", weight: 1, cost: 1000}

  describe "find/1" do
    test "three items set" do
      items = [@guitar, @stereo, @laptop]

      assert Enum.sort([@laptop, @guitar]) == Enum.sort(Backpack.find(items))
    end

    test "three items set with different order" do
      items = [@laptop, @guitar, @stereo]

      assert Enum.sort([@laptop, @guitar]) == Enum.sort(Backpack.find(items))
    end

    test "four items set" do
      items = [@iphone, @guitar, @stereo, @laptop]

      assert Enum.sort([@laptop, @iphone]) == Enum.sort(Backpack.find(items))
    end

    test "five items set" do
      items = [@iphone, @guitar, @stereo, @laptop, @mp3]

      assert Enum.sort([@laptop, @iphone]) == Enum.sort(Backpack.find(items))
    end
  end
end
