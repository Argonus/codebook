defmodule Pooly.Server do
  @moduledoc false

  defmodule State do
    defstruct sup: nil, size: nil, mfa: nil, worker_sup: nil, workers: [], monitors: nil
  end

  use GenServer
  import Supervisor.Spec

  #######
  # API #
  #######

  def start_link(sup, pool_config) do
    GenServer.start_link(__MODULE__, [sup, pool_config], name: __MODULE__)
  end

  def checkout do
    GenServer.call(__MODULE__, :checkout)
  end

  def checkin(worker_pid) do
    GenServer.cast(__MODULE__, {:checkin, worker_pid})
  end

  def status do
    GenServer.call(__MODULE__, :status)
  end

  #############
  # Callbacks #
  #############

  def init([sup, pool_config]) when is_pid(sup), do: init(pool_config, %State{sup: sup})
  def init([{:mfa, mfa} | rest], state), do: init(rest, %State{state | mfa: mfa})
  def init([{:size, size} | rest], state), do: init(rest, %State{state | size: size})
  def init([], state) do
    monitors = :ets.new(:monitors, [:private])
    send(self(), :start_worker_supervisor)
    {:ok, %State{state | monitors: monitors}}
  end

  def handle_info(:start_worker_supervisor, state = %State{sup: sup, mfa: mfa, size: size}) do
    {:ok, worker_sup} = Supervisor.start_child(sup, supervisor_spec(mfa))
    workers = pre_populate(size, worker_sup)
    {:noreply, %State{state | worker_sup: worker_sup, workers: workers}}
  end

  def handle_call(
        :checkout,
        {from_pid, _ref},
        state = %State{workers: workers, monitors: monitors}
      ) do
    case workers do
      [worker | rest] ->
        ref = Process.monitor(from_pid)
        true = :ets.insert(monitors, {worker, ref})
        {:reply, worker, %State{state | workers: rest}}

      [] ->
        {:reply, :noproc, state}
    end
  end

  def handle_call(:status, _from, state = %State{workers: workers, monitors: monitors}) do
    {:reply, {length(workers), :ets.info(monitors)}, state}
  end

  def handle_cast({:checkin, worker_pid}, state = %State{workers: workers, monitors: monitors}) do
    case :ets.lookup(:monitors, worker_pid) do
      [{pid, ref}] ->
        true = Process.demonitor(ref)
        true = :ets.delete(monitors, pid)
        {:noreply, %State{state | workers: [worker_pid | workers]}}

      [] ->
        {:norpley, state}
    end
  end

  #####################
  # Private Functions #
  #####################

  defp supervisor_spec(mfa) do
    opts = [restart: :temporary]
    supervisor(Pooly.WorkerSupervisor, [mfa], opts)
  end

  defp pre_populate(size, worker_sup), do: pre_populate(size, worker_sup, [])

  defp pre_populate(size, _worker_sup, workers) when size < 1, do: workers

  defp pre_populate(size, worker_sup, workers) do
    pre_populate(size - 1, worker_sup, [new_worker(worker_sup) | workers])
  end

  def new_worker(worker_sup) do
    {:ok, worker} = Supervisor.start_child(worker_sup, [[]])
    worker
  end
end
