defmodule CacheServer.Worker do
  @moduledoc false

  use GenServer

  require Logger

  ## Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def write(pid_or_name, key, value), do: GenServer.call(pid_or_name, {:write, key, value})

  def read(pid_or_name, key), do: GenServer.call(pid_or_name, {:read, key})

  def delete(pid_or_name, key), do: GenServer.cast(pid_or_name, {:delete, key})

  def clear(pid_or_name), do: GenServer.cast(pid_or_name, :clear)

  def exists?(pid_or_name, key), do: GenServer.call(pid_or_name, {:exists, key})

  def stop(pid_or_name), do: GenServer.cast(pid_or_name, :stop)

  ## GenServer API

  def init(_args), do: {:ok, %{}}

  def handle_call({:write, key, value}, _from, state) do
    new_state = Map.put(state, key, value)
    {:reply, :ok, new_state}
  end

  def handle_call({:read, key}, _from, state) do
    return_value =
      case Map.get(state, key) do
        nil -> {:error, :not_found}
        value -> {:ok, value}
      end

    {:reply, return_value, state}
  end

  def handle_call({:exists, key}, _from, state) do
    if Map.has_key?(state, key) do
      {:reply, true, state}
    else
      {:reply, false, state}
    end
  end

  def handle_cast({:delete, key}, state), do: {:noreply, Map.delete(state, key)}
  def handle_cast(:clear, _state), do: {:noreply, %{}}
  def handle_cast(:stop, state), do: {:stop, :normal, state}

  def handle_info(msg, state) do
    Logger.warn("Received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  def terminate(reason, state) do
    Logger.info(
      "Server terminated because of #{inspect(reason)}. Last know state: #{inspect(state)}"
    )

    :ok
  end
end
