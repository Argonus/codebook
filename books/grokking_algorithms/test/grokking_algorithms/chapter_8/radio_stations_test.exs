defmodule GrokkingAlgorithms.Chapter8.RadioStationsTest do
  use ExUnit.Case, async: true

  alias GrokkingAlgorithms.Chapter8.RadioStations

  describe "find_station_list/1" do
    test "finds most optimal stations list" do
      stations_list =
        [
          %{name: :k_one, states: ["ID", "NV", "UT"]},
          %{name: :k_two, states: ["WA", "ID", "MT"]},
          %{name: :k_three, states: ["OR", "NV", "CA"]},
          %{name: :k_four, states: ["NV", "UT"]},
          %{name: :k_five, states: ["CA", "AZ"]}
        ]
        |> Enum.shuffle()

      assert Enum.sort([:k_one, :k_two, :k_three, :k_five]) ==
               Enum.sort(RadioStations.find_station_list(stations_list))
    end
  end
end
