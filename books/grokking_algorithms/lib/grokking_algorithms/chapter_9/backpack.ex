defmodule GrokkingAlgorithms.Chapter9.Backpack do
  @moduledoc false

  @backpack_size 4

  @type item :: %{name: String.t(), weight: integer, cost: integer}

  @spec find([item]) :: [item]
  def find(items) do
    num_of_columns = find_num_of_columns(items)

    items
    |> Enum.with_index()
    |> Enum.reduce([], &fill_matrix(&1, &2, num_of_columns))
    |> Enum.flat_map(& &1)
    |> Enum.max_by(&fetch_cost/1)
    |> fetch_items()
  end

  defp find_num_of_columns(items) do
    weights = (Enum.map(items, & &1.weight) ++ [@backpack_size]) |> Enum.uniq()
    weights_gcd = greatest_common_divisor(weights)

    (@backpack_size / weights_gcd) |> trunc()
  end

  defp greatest_common_divisor([h1]), do: h1

  defp greatest_common_divisor([h1, h2 | t]) do
    greatest_common_divisor([Integer.gcd(h1, h2) | t])
  end

  defp fill_matrix({item, row_index}, acc, num_of_columns) do
    row =
      Enum.map(0..(num_of_columns - 1), fn col_index ->
        find_max_value(item, row_index, col_index, acc)
      end)

    [row | acc]
  end

  defp find_max_value(item, 0, _, _), do: {item.cost, [item]}

  defp find_max_value(item, row_index, col_index, acc) do
    # previous row values
    prev_row = Enum.fetch!(acc, row_index - 1)
    prev_value = {prev_cost, _} = Enum.fetch!(prev_row, col_index)

    # current row value
    current_cost = item.cost
    free_space = col_index - weight_to_index(item.weight)
    {free_space_cost, items} = find_items_for_free_space(free_space, prev_row)
    maybe_new_cost = current_cost + free_space_cost

    cond do
      # Not enough space
      col_index < weight_to_index(item.weight) -> prev_value
      # New item
      maybe_new_cost > prev_cost -> {maybe_new_cost, [item | items]}
      # Prev item
      true -> prev_value
    end
  end

  defp weight_to_index(weight), do: weight - 1

  defp find_items_for_free_space(free_space, _prev_row) when free_space <= 0, do: {0, []}
  defp find_items_for_free_space(free_space, prev_row), do: Enum.fetch!(prev_row, free_space)

  defp fetch_cost({cost, _items}), do: cost

  defp fetch_items({_cost, items}), do: items
end
