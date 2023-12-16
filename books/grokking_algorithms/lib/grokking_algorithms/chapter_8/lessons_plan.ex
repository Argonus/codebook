defmodule GrokkingAlgorithms.Chapter8.LessonsPlan do
  @moduledoc false

  @spec find_plan(list) :: list
  def find_plan(lessons) do
    lessons
    |> Enum.sort_by(& &1.start_time)
    |> Enum.reduce([], &add_to_class(&1, &2))
  end

  defp add_to_class(lesson, []), do: [lesson]

  defp add_to_class(lesson, [last_lesson | _t] = acc) do
    if lesson.start_time >= last_lesson.end_time do
      [lesson | acc]
    else
      acc
    end
  end
end
