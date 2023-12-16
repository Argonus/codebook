defmodule GrokkingAlgorithms.Chapter10.RecommendationEngineTest do
  use ExUnit.Case, async: true

  alias GrokkingAlgorithms.Chapter10.RecommendationEngine

  describe "find_nearest/1" do
    test "find nearest for me" do
      person =
        {:peter,
         %{
           comedy: 3,
           action: 4,
           drama: 2,
           horror: 1,
           romance: 2
         }}

      [{nearest_name, _} | _] = RecommendationEngine.find_nearest(person)

      assert nearest_name == :justyne
    end
  end
end
