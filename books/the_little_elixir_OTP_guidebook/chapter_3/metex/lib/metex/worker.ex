defmodule Metex.Worker do
  @moduledoc false

  require Logger

  @open_weather_base_url "http://api.openweathermap.org"
  @open_weather_weather_endpoint "data/2.5/weather?q="
  @open_weather_api_key "API_KEY"
  @open_weather_app_id_param "&appid=#{@open_weather_api_key}"
  @kelvin_celsius_diff 273.15

  def loop do
    receive do
      {sender, location} ->
        send(sender, {:ok, temperature_of(location)})

      message ->
        Logger.info("unexpected message #{message}")
    end
  end

  defp temperature_of(location) do
    case fetch_temperature_from_api(location) do
      {:ok, temp} ->
        "#{location}: #{temp}"

      {:error, :not_found} ->
        "#{location}: location not found"

      :error ->
        "Something went wrong"
    end
  end

  defp fetch_temperature_from_api(location) do
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
