defmodule Pooly.Server do
  @moduledoc false

  use GenServer
  import Supervisor.Spec

  #######
  # API #
  #######

  def start_link(pools_config) do
    GenServer.start_link(__MODULE__, pools_config, name: __MODULE__)
  end

  @doc false
  def checkout(pool_name, block, timeout) do
    Pooly.PoolServer.checkout(pool_name, block, timeout)
  end

  @doc false
  def checkin(pool_name, worker_pid),
    do: GenServer.cast(server_name(pool_name), {:checkin, worker_pid})

  @doc false
  def status(pool_name), do: GenServer.call(server_name(pool_name), :status)

  #############
  # Callbacks #
  #############

  def init(pools_config) do
    Enum.each(pools_config, fn pool ->
      send(self(), {:start_pool, pool})
    end)

    {:ok, pools_config}
  end

  def handle_info({:start_pool, pool}, state) do
    {:ok, _pool_sup} = Supervisor.start_child(Pooly.PoolsSupervisor, supervisor_spec(pool))
    {:noreply, state}
  end

  #####################
  # Private Functions #
  #####################

  defp supervisor_spec(pool_config) do
    opts = [id: supervisor_name(pool_config[:name])]
    supervisor(Pooly.PoolSupervisor, [pool_config], opts)
  end

  defp server_name(pool_name), do: :"#{pool_name}Server"
  defp supervisor_name(pool_name), do: :"#{pool_name}Supervisor"
end
