defmodule GrokkingAlgorithms.Chapter4.QuickSort do
  @moduledoc false

  @spec sort(list) :: list
  def sort([]), do: []

  def sort([pivot | tail]) do
    {smaller, bigger} = Enum.split_with(tail, &(&1 < pivot))
    sort(smaller) ++ [pivot] ++ sort(bigger)
  end
end
