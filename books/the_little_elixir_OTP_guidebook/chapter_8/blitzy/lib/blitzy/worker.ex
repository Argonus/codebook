defmodule Blitzy.Worker do
  @moduledoc false
  require Logger

  def start(url) do
    {timestamp, http_response} =
      Timex.Duration.measure(fn ->
        HTTPoison.get(url)
      end)

    milliseconds = Timex.Duration.to_milliseconds(timestamp)
    handle_response(milliseconds, http_response)
  end

  defp handle_response(milliseconds, {:ok, %HTTPoison.Response{status_code: code}})
       when code >= 200 and code <= 304 do
    Logger.info("Worker #{worker_signature()} completed in #{milliseconds}")
    {:ok, milliseconds}
  end

  defp handle_response(_milliseconds, {:error, reason}) do
    Logger.info("Worker #{worker_signature()} error due to #{inspect(reason)}")
    {:error, reason}
  end

  defp handle_response(_, _) do
    Logger.info("Worker #{worker_signature()} errored out")
    {:error, :unknown}
  end

  defp worker_signature do
    "[#{inspect(node())}-#{inspect(self())}]"
  end
end
