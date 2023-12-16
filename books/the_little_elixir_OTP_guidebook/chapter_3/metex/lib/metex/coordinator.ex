defmodule Metex.Coordinator do
  @moduledoc false

  require Logger

  def loop(results \\ [], results_expected) do
    receive do
      {:ok, result} ->
        new_results = [result | results]
        if results_expected == Enum.count(new_results) do
          send(self(), :exit)
        end
        loop(new_results, results_expected)
      :exit ->
        formatted_results = format_results(results)
        Logger.info(formatted_results)
      _ ->
        loop(results, results_expected)
    end
  end

  defp format_results(results) do
    results
    |> Enum.sort
    |> Enum.join(", ")
  end
end