defmodule GrokkingAlgorithms.Chapter9.LondonTripTest do
  use ExUnit.Case, async: true

  alias GrokkingAlgorithms.Chapter9.LondonTrip

  @westminster %{name: "Westminster", time: 12, points: 7}
  @globe_theatre %{name: "Globe Theater", time: 12, points: 6}
  @national_gallery %{name: "National Gallery", time: 24, points: 9}
  @british_museum %{name: "British Museum", time: 48, points: 9}
  @st_paul_cathedral %{name: "St Paul Cathedral", time: 12, points: 8}

  describe "find/1" do
    test "five items set" do
      items = [
        @westminster,
        @globe_theatre,
        @national_gallery,
        @british_museum,
        @st_paul_cathedral
      ]

      assert Enum.sort([@westminster, @national_gallery, @st_paul_cathedral]) ==
               Enum.sort(LondonTrip.find(items))
    end
  end
end
