defmodule GrokkingAlgorithms.Chapter4.Length do
  @moduledoc false

  @spec length(list) :: integer
  def length([]), do: 0
  def length([_]), do: 1
  def length(list) when is_list(list), do: length_rec(list, 0)

  defp length_rec([_], acc), do: acc + 1
  defp length_rec([_ | h], acc), do: length_rec(h, acc + 1)
end
