use Mix.Config

defmodule Blitzy.CLI do
  require Logger

  alias Blitzy.TasksSupervisor

  def main(args) do
    master = Application.get_env(:blitzy, :master_node)
    slaves = Application.get_env(:blitzy, :slave_nodes)

    Node.start(master)
    Enum.map(slaves, &Node.connect(&1))

    args
    |> parse_args()
    |> process_options([node() | Node.list()])
  end

  defp parse_args(args) do
    OptionParser.parse(args,
      aliases: [n: :requests],
      strict: [requests: :integer]
    )
  end

  defp process_options(options, nodes) do
    case options do
      {[requests: n], [url], _} ->
        do_requests(n, url, nodes)

      _ ->
        do_help()
    end
  end

  defp do_help() do
    IO.puts("
    Usage:
    blitzy -n [requests] [url]
    -n, --requests # number of requests

    Example
    blitzy -n 100 http://www.bieberfever.com/
    ")
  end

  defp do_requests(n, url, nodes) do
    Logger.info("Pummeling #{url} with #{n} requests")

    total_nodes = Enum.count(nodes)
    req_per_node = div(n, total_nodes)

    nodes
    |> Enum.flat_map(fn node ->
      1..req_per_node
      |> Enum.map(fn _ ->
        Task.Supervisor.async({TasksSupervisor, node}, Blitzy.Worker, :start, [url])
      end)
      |> Enum.map(& Task.await(&1, :infinity))
      |> parse_results()
    end)
  end

  defp parse_results(results) do
    {successes, _failures} =
      Enum.split_with(results, fn result ->
        case result do
          {:ok, _} -> true
          _ -> false
        end
      end)

    total_workers = Enum.count(results)
    total_success = Enum.count(successes)
    total_failures = total_workers - total_success

    data = Enum.map(successes, fn {_, time} -> time end)

    average_time = average(data)
    longest_time = Enum.max(data)
    shortest_time = Enum.min(data)

    Logger.info("
    Total Workers: #{total_workers}
    Total Successes: #{total_success}
    Total Failures: #{total_failures}
    Average (msecs): #{average_time}
    Longest Time (msecs): #{longest_time}
    Shortest Time (msecs): #{shortest_time}
    ")
    results
  end

  def average(data) do
    sum = Enum.sum(data)

    if sum > 0 do
      sum / Enum.count(data)
    else
      0
    end
  end
end
