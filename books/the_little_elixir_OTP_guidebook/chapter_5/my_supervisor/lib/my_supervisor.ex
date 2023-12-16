defmodule MySupervisor do
  @moduledoc """
  Documentation for MySupervisor.
  """

  use GenServer

  #######
  # API #
  #######

  def start_link(child_spec_list) do
    GenServer.start_link(__MODULE__, [child_spec_list])
  end

  def start_child(pid, child_spec) do
    GenServer.call(pid, {:start_child, child_spec})
  end

  def terminate_child(pid, child_pid) when is_pid(child_pid) do
    GenServer.call(pid, {:terminate, child_pid})
  end

  def restart_child(pid, child_pid, child_spec) when is_pid(child_pid) do
    GenServer.call(pid, {:restart_child, child_pid, child_spec})
  end

  def count_children(pid), do: GenServer.call(pid, :count_children)

  def which_children(pid), do: GenServer.call(pid, :which_children)

  #######################
  # GenServer CALLBACKS #
  #######################

  def init([child_spec_list]) do
    Process.flag(:trap_exit, true)

    state =
      child_spec_list
      |> start_children()
      |> Enum.into(Map.new())

    {:ok, state}
  end

  def handle_call({:start_child, child_spec}, _from, state) do
    case start_child(child_spec) do
      {:ok, pid} ->
        new_state = Map.put(state, pid, child_spec)
        {:reply, {:ok, pid}, new_state}

      :error ->
        {:reply, {:error, "error with starting child"}, state}
    end
  end

  def handle_call({:restart_child, old_pid, _child_spec}, _from, state) do
    case Map.get(state, old_pid) do
      nil ->
        {:reply, {:error, :not_found}, state}

      child_spec ->
        case restart_child(old_pid, child_spec) do
          {:ok, {new_pid, child_spec}} ->
            new_state =
              state
              |> Map.delete(old_pid)
              |> Map.put(new_pid, child_spec)

            {:reply, :ok, new_state}

          :error ->
            {:reply, {:error, "error restarting child"}, state}
        end
    end
  end

  def handle_call(:count_children, _from, state) do
    {:reply, length(state), state}
  end

  def handle_call(:which_children, _from, state) do
    {:reply, state, state}
  end

  def handle_call({:terminate, pid}, _from, state) do
    case terminate_child(pid) do
      :ok ->
        new_state = Map.delete(state, pid)
        {:reply, :ok, new_state}

      :error ->
        {:reply, {:error, "error terminating child"}, state}
    end
  end

  def handle_info({:EXIT, from, :killed}, state) do
    new_state = Map.delete(state, from)
    {:noreply, new_state}
  end

  def handle_info({:EXIT, from, :normal}, state) do
    new_state = Map.delete(state, from)
    {:noreply, new_state}
  end

  def handle_info({:EXIT, old_pid, _reason}, state) do
    case Map.get(state, old_pid) do
      nil ->
        {:noreply, state}

      child_spec ->
        case restart_child(old_pid, child_spec) do
          {:ok, {new_pid, child_spec}} ->
            new_state =
              state
              |> Map.delete(old_pid)
              |> Map.put(new_pid, child_spec)

            {:noreply, new_state}

          :error ->
            {:noreply, state}
        end
    end
  end

  def terminate(_reason, state) do
    terminate_children(state)
    :ok
  end

  ####################
  # Helper Functions #
  ####################

  defp start_children([]), do: []

  defp start_children([child_spec | rest]) do
    case start_child(child_spec) do
      {:ok, pid} ->
        [{pid, child_spec} | start_children(rest)]

      :error ->
        :error
    end
  end

  defp start_child({mod, fun, args}) do
    case apply(mod, fun, args) do
      pid when is_pid(pid) ->
        Process.link(pid)
        {:ok, pid}

      _ ->
        :error
    end
  end

  defp restart_child(old_pid, child_spec) do
    case terminate_child(old_pid) do
      :ok ->
        case start_child(child_spec) do
          {:ok, new_pid} -> {:ok, {new_pid, child_spec}}
          :error -> :error
        end

      :error ->
        :error
    end
  end

  defp terminate_children(children) do
    Enum.each(children, fn {pid, _} ->
      terminate_child(pid)
    end)
  end

  defp terminate_child(pid) do
    Process.exit(pid, :kill)
    :ok
  end
end
