defmodule Chapter2.Exercise1 do
  @moduledoc false

  @doc """
  Hello world.

  ## Examples

      iex> Chapter2.Exercise1.rec_sum([1, 2])
      3

  """
  @spec rec_sum([integer]) :: integer
  def rec_sum([]), do: 0
  def rec_sum([head | tail]) when is_integer(head), do: head + rec_sum(tail)

  @doc """
  Hello world.

  ## Examples

      iex> Chapter2.Exercise1.tail_rec_sum([1, 2])
      3

  """
  @spec tail_rec_sum([integer]) :: integer
  def tail_rec_sum(list), do: do_tail_rec_sum(list, 0)

  defp do_tail_rec_sum([], acc), do: acc
  defp do_tail_rec_sum([head], acc), do: acc + head
  defp do_tail_rec_sum([head | tail], acc), do: do_tail_rec_sum(tail, acc + head)
end
