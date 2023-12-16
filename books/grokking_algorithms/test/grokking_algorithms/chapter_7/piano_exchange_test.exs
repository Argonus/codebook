defmodule GrokkingAlgorithms.Chapter7.PianoExchangeTest do
  use ExUnit.Case, async: true

  alias GrokkingAlgorithms.Chapter7.PianoExchange

  describe "find_cost/1" do
    test "returns cost for searched node when it's a start" do
      assert {:ok, 0} = PianoExchange.find_cost(:piano)
    end

    test "returns cost for searched node when it's not a start" do
      assert {:ok, 35} = PianoExchange.find_cost(:book)
    end
  end

  describe "find_path/1" do
    test "returns cost for searched node when it's a start" do
      assert {:ok, []} = PianoExchange.find_path(:piano)
    end

    test "returns cost for searched node when it's not a start" do
      assert {:ok, [:cd, :percussion, :piano]} = PianoExchange.find_path(:book)
    end
  end
end
