defmodule Metex.Worker do
  @moduledoc false

  use GenServer

  require Logger

  @open_weather_base_url "http://api.openweathermap.org"
  @open_weather_weather_endpoint "data/2.5/weather?q="
  @open_weather_api_key "API_KEY"
  @open_weather_app_id_param "&appid=#{@open_weather_api_key}"
  @kelvin_celsius_diff 273.15

  ## Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_temperature(pid, location) do
    GenServer.call(pid, {:location, location})
  end

  def get_stats(pid), do: GenServer.call(pid, :get_stats)

  def reset_stats(pid), do: GenServer.cast(pid, :reset_stats)

  def stop(pid), do: GenServer.cast(pid, :stop)

  ## GenServer API

  def init(_args), do: {:ok, %{}}

  def handle_call({:location, location}, _from, state) do
    case temperature_of(location) do
      {:ok, temp} ->
        new_state = update_state(state, location)
        {:reply, "#{temp} C", new_state}
      _ ->
        {:reply, :error, state}
    end
  end

  def handle_call(:get_stats, _from, state) do
    {:reply, state, state}
  end

  def handle_cast(:reset_stats, _state), do: {:noreply, %{}}
  def handle_cast(:stop, state), do: {:stop, :normal, state}

  def handle_info(msg, state) do
    Logger.warn("Received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  def terminate(reason, state) do
    Logger.info("Server terminated because of #{inspect(reason)}. Last know stats: #{inspect(state)}")
    :ok
  end

  ## Helper functions

  defp update_state(old_state, location) do
    if Map.has_key?(old_state, location) do
      Map.update!(old_state, location, &(&1 + 1))
    else
      Map.put_new(old_state, location, 1)
    end
  end

  defp temperature_of(location) do
    location
    |> url_for()
    |> HTTPoison.get()
    |> parse_response()
  end

  defp url_for(location) do
    encoded_location = URI.encode(location)

    "#{@open_weather_base_url}/#{@open_weather_weather_endpoint}#{encoded_location}#{
      @open_weather_app_id_param
    }"
  end

  defp parse_response({:ok, %HTTPoison.Response{body: body, status_code: 200}}) do
    body
    |> Jason.decode()
    |> compute_temperature()
  end

  defp parse_response(_response), do: {:error, :not_found}

  defp compute_temperature({:ok, json_data}) do
    try do
      temp =
        json_data
        |> get_in(["main", "temp"])
        |> calculate_celsius()
        |> Float.round(1)

      {:ok, temp}
    rescue
      _ -> :error
    end
  end

  defp calculate_celsius(temp), do: temp - @kelvin_celsius_diff
end
