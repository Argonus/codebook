defmodule Metex do
  @moduledoc """
  Documentation for Metex.
  """

  def temperature_of(locations) do
    coordinator_pid = spawn(Metex.Coordinator, :loop, [[], Enum.count(locations)])

    locations
    |> Enum.each(
         fn city ->
           worker_pid = spawn(Metex.Worker, :loop, [])
           send(worker_pid, {coordinator_pid, city})
         end
       )
  end
end
