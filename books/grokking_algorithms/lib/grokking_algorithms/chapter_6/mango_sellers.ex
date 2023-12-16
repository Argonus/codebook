defmodule GrokkingAlgorithms.Chapter6.MangoSellers do
  @moduledoc false

  @graph %{
    piotr: [:alicja, :bartek, :cecylia],
    bartek: [:janusz, :patrycja],
    alicja: [:patrycja],
    cecylia: [:tamara, :jarek],
    janusz: [],
    patrycja: [],
    tamara: [],
    jarek: []
  }

  def mango_seller(name, is_seller_fn) when is_binary(name) do
    name
    |> String.to_existing_atom()
    |> mango_seller(is_seller_fn)
  end

  def mango_seller(name, is_seller_fn) do
    if is_seller_fn.(name) do
      {:ok, name}
    else
      name
      |> queue_from_name()
      |> check_queue([], is_seller_fn)
    end
  end

  defp check_queue(queue, searched, is_seller_fn) do
    queue
    |> :queue.out()
    |> check_seller(searched, is_seller_fn)
  end

  defp queue_from_name(name) do
    @graph
    |> Map.get(name)
    |> :queue.from_list()
  end

  defp check_seller({:empty, _}, _, _), do: {:error, :not_found}

  defp check_seller({{:value, name}, queue}, searched, is_seller_fn) do
    cond do
      is_seller_fn.(name) ->
        {:ok, name}

      Enum.member?(searched, name) ->
        check_queue(queue, searched, is_seller_fn)

      true ->
        name
        |> add_new_to_queue(queue)
        |> check_queue(searched, is_seller_fn)
    end
  end

  defp add_new_to_queue(name, queue) do
    queue_2 = queue_from_name(name)
    :queue.join(queue, queue_2)
  end
end
