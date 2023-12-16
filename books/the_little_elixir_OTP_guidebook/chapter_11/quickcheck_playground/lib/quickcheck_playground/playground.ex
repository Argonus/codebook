defmodule QuickcheckPlayground.Playground do
  @moduledoc false

  def is_sorted?([]), do: true

  def is_sorted?(list) do
    list
    |> Enum.zip(tl(list))
    |> Enum.all?(fn {x, y} -> x <= y end)
  end

  def model_store(k, v, list) do
    case find_index_with_key(k, list) do
      {:match, index} ->
        List.replace_at(list, index, {k, v})

      _ ->
        [{k, v} | list]
    end
  end

  def find_index_with_key(k, list) do
    case Enum.find_index(list, fn {x, _} -> x == k end) do
      nil -> :nomatch
      index -> {:match, index}
    end
  end
end
