defmodule GrokkingAlgorithms.Chapter8.RadioStations do
  @moduledoc false

  @spec find_station_list(list) :: list
  def find_station_list(stations) do
    stations
    |> Enum.map(&%{name: &1.name, states: MapSet.new(&1.states)})
    |> Enum.sort_by(&MapSet.size(&1.states), :desc)
    |> build_state()
    |> find_stations()
    |> MapSet.to_list()
  end

  defp build_state(stations) do
    states_needed =
      stations
      |> Enum.reduce(MapSet.new([]), &MapSet.union(&1.states, &2))

    %{
      stations: stations,
      selected_stations: MapSet.new([]),
      states_needed: states_needed
    }
  end

  defp find_stations(stations_data) do
    case stations_data.states_needed do
      %MapSet{map: map} when map_size(map) == 0 ->
        stations_data.selected_stations

      states_needed ->
        stations = stations_data.stations
        new_station = find_best_station(stations, nil, states_needed, MapSet.new([]))
        new_states_needed = MapSet.difference(states_needed, new_station.states)

        new_stations_data = %{
          stations: List.delete(stations, new_station),
          selected_stations: MapSet.put(stations_data.selected_stations, new_station.name),
          states_needed: new_states_needed
        }

        find_stations(new_stations_data)
    end
  end

  defp find_best_station([], station, _, _), do: station
  defp find_best_station([station], nil, _, _), do: station

  defp find_best_station([station | t], maybe_best, states_needed, states_covered) do
    new_states_covered = MapSet.intersection(station.states, states_needed)

    if MapSet.size(new_states_covered) > MapSet.size(states_covered) do
      find_best_station(t, station, states_needed, new_states_covered)
    else
      find_best_station(t, maybe_best, states_needed, states_covered)
    end
  end
end
