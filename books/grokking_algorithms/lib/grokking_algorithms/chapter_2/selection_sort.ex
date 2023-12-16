defmodule GrokkingAlgorithms.Chapter2.SelectionSort do
  @moduledoc false

  def sort([]), do: []
  def sort([el]), do: [el]
  def sort(list) when is_list(list), do: do_sort(list, [])

  def do_sort([], acc), do: acc

  def do_sort(list, acc) do
    max = Enum.max(list)
    new_list = List.delete(list, max)
    do_sort(new_list, [max | acc])
  end
end
