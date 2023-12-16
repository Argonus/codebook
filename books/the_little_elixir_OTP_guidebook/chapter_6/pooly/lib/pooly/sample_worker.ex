defmodule Pooly.SampleWorker do
  @moduledoc false

  use GenServer

  def start_link(_) do
    GenServer.start_link(__MODULE__, [])
  end

  def stop(pid) do
    GenServer.call(:stop, pid)
  end

  def handle_call(:stop, _from, state) do
    {:stop, :normal, :ok, state}
  end
end
