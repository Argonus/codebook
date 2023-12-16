defmodule Ring do
  @moduledoc """
  Documentation for Ring.
  """

  def create_processes(n) do
    Enum.map(
      1..n,
      fn _n ->
        spawn(&loop/0)
      end
    )
  end

  def create_and_link_processes(n) do
    Enum.map(
      1..n,
      fn _n ->
        spawn_link(&loop/0)
      end
    )
  end

  def check_processes(pids) do
    Enum.map(
      pids,
      fn pid ->
        IO.inspect(
          "pid: #{inspect(pid)} alive #{inspect(Process.alive?(pid))} -> links #{inspect(Process.info(pid, :links))}"
        )
      end
    )
  end

  def kill_random(pids) do
    pids
    |> Enum.shuffle()
    |> List.first()
    |> send(:crash)
  end

  def link_processes(procs) do
    link_processes(procs, [])
  end

  defp link_processes([], _), do: :ok

  defp link_processes([proc | []], linked_processes)do
    first_process = List.last(linked_processes)
    send(proc, {:link, first_process})
    :ok
  end

  defp link_processes([proc_1, proc_2 | rest], linked_processes) do
    send(proc_1, {:link, proc_2})
    link_processes([proc_2 | rest], [proc_1 | linked_processes])
  end

  defp loop do
    receive do
      {:link, link_to} when is_pid(link_to) ->
        Process.link(link_to)
        loop()
      :trap_exit ->
        Process.flag(:trap_exit, true)
        loop()
      {:EXIT, pid, reason} ->
        IO.inspect("#{inspect(self())} received {:EXIT, #{inspect(pid)}, #{reason}")
        loop()
      :crash ->
        1 / 0
    end
  end
end
