defmodule GrokkingAlgorithms.Chapter8.LessonsPlanTest do
  use ExUnit.Case, async: true

  alias GrokkingAlgorithms.Chapter8.LessonsPlan

  describe "find_plan/1" do
    test "find best plan for one class" do
      lessons_list =
        [
          %{subject: "crafts", start_time: 9.00, end_time: 10.00},
          %{subject: "english", start_time: 9.30, end_time: 10.30},
          %{subject: "math", start_time: 10.00, end_time: 11.00},
          %{subject: "computer_science", start_time: 10.30, end_time: 11.30},
          %{subject: "music", start_time: 11.00, end_time: 12.00}
        ]
        |> Enum.shuffle()

      assert [
               %{subject: "music", start_time: 11.00, end_time: 12.00},
               %{subject: "math", start_time: 10.00, end_time: 11.00},
               %{subject: "crafts", start_time: 9.00, end_time: 10.00}
             ] == LessonsPlan.find_plan(lessons_list)
    end
  end
end
