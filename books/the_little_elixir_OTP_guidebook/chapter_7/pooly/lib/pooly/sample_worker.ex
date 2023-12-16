defmodule Pooly.SampleWorker do
  @moduledoc false

  use GenServer

  def start_link(_) do
    GenServer.start_link(__MODULE__, [])
  end

  def work_for(pid, duration) do
    GenServer.cast(pid, {:work_for, duration})
  end

  def work(pid, fun) do
    GenServer.call(pid, {:work, fun})
  end

  def init(init_arg) do
    {:ok, init_arg}
  end

  def stop(pid) do
    GenServer.call(:stop, pid)
  end

  def handle_call(:stop, _from, state) do
    {:stop, :normal, :ok, state}
  end

  def handle_call({:work, fun}, _from, state) do
    fun.()
    {:reply, :ok, state}
  end

  def handle_cast({:work_for, duration}, state) do
    :timer.sleep(duration)
    {:stop, :normal, state}
  end
end
