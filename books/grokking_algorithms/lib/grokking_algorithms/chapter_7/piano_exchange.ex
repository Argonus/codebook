defmodule GrokkingAlgorithms.Chapter7.PianoExchange do
  @moduledoc false

  @graph %{
    book: %{cd: 5, poster: 0},
    cd: %{guitar: 15, percussion: 20},
    poster: %{guitar: 30, percussion: 35},
    guitar: %{piano: 20},
    percussion: %{piano: 10},
    piano: %{}
  }

  @spec find_cost(atom) :: {:ok, integer}
  def find_cost(:piano), do: {:ok, 0}

  def find_cost(start_name) do
    # build_init_state
    costs = build_init_costs(start_name)
    parents = build_init_parents(start_name)

    {:ok, costs, _parents, _processed} = do_find(costs, parents, [start_name], 0)
    cost = Map.get(costs, :piano)
    #
    {:ok, cost}
  end

  def find_path(:piano), do: {:ok, []}

  def find_path(name) do
    costs = build_init_costs(name)
    parents = build_init_parents(name)

    {:ok, _costs, parents, _processed} = do_find(costs, parents, [name], 0)
    path = do_build_path(parents, :piano, [])

    {:ok, path}
  end

  defp do_build_path(parents, search, path) do
    case Map.get(parents, search) do
      nil ->
        path

      parent ->
        new_parents = Map.delete(parents, search)
        new_path = [search | path]
        do_build_path(new_parents, parent, new_path)
    end
  end

  defp build_init_costs(name) do
    basic_costs = Map.get(@graph, name)

    all_costs =
      @graph
      |> Map.keys()
      |> Enum.reject(&(&1 == name))
      |> Enum.reduce(%{}, fn n, acc ->
        Map.merge(acc, %{n => :infinity})
      end)

    Map.merge(all_costs, basic_costs)
  end

  defp build_init_parents(name) do
    node_parents = @graph |> Map.get(name) |> Map.keys()

    @graph
    |> Map.keys()
    |> Enum.reject(&(&1 == name))
    |> Enum.reduce(%{}, fn n, acc ->
      parent = if Enum.member?(node_parents, n), do: name, else: nil
      Map.merge(acc, %{n => parent})
    end)
  end

  defp do_find(costs, parents, processed, step) do
    case find_lowest_node(costs, processed) do
      [{node, _v} | _tail] ->
        {new_costs, new_parents} = update_state(node, costs, parents)
        new_processed = [node | processed]
        do_find(new_costs, new_parents, new_processed, step + 1)

      _ ->
        {:ok, costs, parents, processed}
    end
  end

  defp update_state(node, costs, parents) do
    cost = Map.get(costs, node)
    neighbours = Map.get(@graph, node)

    Enum.reduce(neighbours, {costs, parents}, fn {n, n_cost}, {acc_costs, acc_parents} = acc ->
      current_cost = Map.get(costs, n)
      new_cost = n_cost + cost

      if current_cost == :infinity || new_cost < current_cost do
        new_costs = Map.put(acc_costs, n, new_cost)
        new_parents = Map.put(acc_parents, n, node)
        {new_costs, new_parents}
      else
        acc
      end
    end)
  end

  defp find_lowest_node(costs, processed) do
    costs
    |> Enum.reject(fn {key, _value} ->
      Enum.member?(processed, key)
    end)
    |> Enum.sort_by(fn {_key, value} ->
      value
    end)
  end
end
