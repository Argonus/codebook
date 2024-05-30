defmodule CheckersWeb.PageController do
  use CheckersWeb, :controller

  alias Checkers.Matches
  alias Checkers.Seasons

  def home(conn, _params) do
    case Pow.Plug.current_user(conn) do
      nil ->
        render(conn, :home)

      _ ->
        {:ok, current_season} = get_current_season()
        matches = get_active_matches(current_season)

        render(conn, :home, current_season: current_season, active_matches: matches)
    end
  end

  defp get_current_season do
    case Seasons.get_current_season() do
      {:error, :not_found} -> Seasons.init_season()
      result -> result
    end
  end

  defp get_active_matches(season) do
    Matches.get_season_matches(season.season_id)
  end
end
