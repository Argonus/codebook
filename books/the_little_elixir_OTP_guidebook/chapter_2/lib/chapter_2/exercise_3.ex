defmodule Chapter2.Exercise3 do
  @moduledoc false
  @list [1, [[2], 3]]

  @spec transform() :: list
  def transform do
    flatten_list = List.flatten(@list)
    reversed_list = Enum.reverse(flatten_list)
    Enum.map(reversed_list, &(&1 * &1))
  end

  @spec pipe_transform() :: list
  def pipe_transform() do
    @list
    |> List.flatten()
    |> Enum.reverse()
    |> Enum.map(&(&1 * &1))
  end
end
