defmodule GrokkingAlgorithms.Chapter4.Max do
  @moduledoc false

  @spec max(list) :: integer
  def max([]), do: nil
  def max([el]), do: el
  def max([h | t]), do: max_rec(t, h)

  defp max_rec([el], acc) when el > acc, do: el
  defp max_rec([_], acc), do: acc

  defp max_rec([t | head], acc) when t > acc, do: max_rec(head, t)
  defp max_rec([_ | head], acc), do: max_rec(head, acc)
end
