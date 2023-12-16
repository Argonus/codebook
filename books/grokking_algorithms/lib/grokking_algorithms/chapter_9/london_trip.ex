defmodule GrokkingAlgorithms.Chapter9.LondonTrip do
  @moduledoc false

  @hours 48

  require IEx

  def find(items) do
    {num_of_columns, weights_gcd} = find_num_of_columns(items)

    items
    |> Enum.map(&reduce_time(&1, weights_gcd))
    |> Enum.with_index()
    |> Enum.reduce(%{}, &fill_matrix(&1, &2, num_of_columns))
    |> Map.values()
    |> Enum.flat_map(& &1)
    |> Enum.max_by(&fetch_cost/1)
    |> fetch_items()
    |> Enum.map(&extend_time(&1, weights_gcd))
  end

  defp find_num_of_columns(items) do
    weights_gcd =
      items
      |> Enum.map(& &1.time)
      |> Kernel.++([@hours])
      |> Enum.uniq()
      |> greatest_common_divisor()

    {(@hours / weights_gcd) |> trunc(), weights_gcd}
  end

  defp greatest_common_divisor([h1]), do: h1

  defp greatest_common_divisor([h1, h2 | t]) do
    greatest_common_divisor([Integer.gcd(h1, h2) | t])
  end

  defp reduce_time(item, weights_gcd) do
    %{item | time: trunc(item.time / weights_gcd)}
  end

  defp fill_matrix({item, row_index}, acc, num_of_columns) do
    row =
      Enum.map(0..(num_of_columns - 1), fn col_index ->
        find_max_value(item, row_index, col_index, acc)
      end)

    Map.put(acc, row_index, row)
  end

  defp find_max_value(item, 0, _, _), do: {item.points, [item]}

  defp find_max_value(item, row_index, col_index, acc) do
    # previous row values
    prev_row = Map.fetch!(acc, row_index - 1)
    prev_value = {prev_points, _} = Enum.fetch!(prev_row, col_index)

    # current row points
    current_points = item.points
    free_time = col_index - time_to_index(item.time)
    {free_time_points, items} = find_items_for_free_time(free_time, prev_row)
    maybe_new_points = current_points + free_time_points

    cond do
      # Not enough space
      col_index < time_to_index(item.time) -> prev_value
      # New item
      maybe_new_points >= prev_points -> {maybe_new_points, items ++ [item]}
      # Prev item
      true -> prev_value
    end
  end

  defp time_to_index(time), do: time - 1

  defp find_items_for_free_time(free_time, _prev_row) when free_time <= 0, do: {0, []}
  defp find_items_for_free_time(free_time, prev_row), do: Enum.fetch!(prev_row, free_time - 1)

  defp fetch_cost({cost, _items}), do: cost

  defp fetch_items({_cost, items}), do: items

  defp extend_time(item, gcd) do
    %{item | time: trunc(item.time * gcd)}
  end
end
