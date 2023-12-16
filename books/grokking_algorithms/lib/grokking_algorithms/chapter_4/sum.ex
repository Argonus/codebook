defmodule GrokkingAlgorithms.Chapter4.Sum do
  @moduledoc false

  @spec sum(list) :: integer
  def sum([]), do: 0
  def sum([el]), do: el
  def sum(list) when is_list(list), do: sum_rec(list, 0)

  defp sum_rec([el], acc), do: acc + el
  defp sum_rec([t | head], acc), do: sum_rec(head, acc + t)
end
