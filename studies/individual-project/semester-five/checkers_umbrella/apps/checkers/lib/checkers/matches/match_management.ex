defmodule Checkers.Matches.MatchManagement do
  @moduledoc false
  import Checkers.Matches.Helpers.MatchHelpers

  alias Checkers.Matches.MatchStruct
  alias Checkers.Repo
  alias Checkers.Schemas.Match, as: MatchSchema

  def get_match(match_id) do
    case Repo.one(match_query(match_id)) do
      nil -> {:error, :not_found}
      match -> {:ok, MatchStruct.build_from_schema(match)}
    end
  end

  defp match_query(match_id) do
    import Ecto.Query, only: [from: 2]

    from(
      m in MatchSchema,
      inner_join: s in assoc(m, :season),
      inner_join: h in assoc(m, :host),
      left_join: p in assoc(m, :player),
      preload: [host: h, player: p, season: s],
      where: m.id == ^match_id
    )
  end

  def get_season_matches(season_id) do
    import Ecto.Query, only: [from: 2]

    from(m in MatchSchema, where: m.season_id == ^season_id)
    |> Repo.all()
    |> Enum.map(&MatchStruct.build_from_schema/1)
  end

  def create_match(host_id) do
    case Checkers.Seasons.get_current_season() do
      {:error, _} ->
        {:error, :no_current_season}

      {:ok, season} ->
        season_id = season.season_id

        host_id
        |> MatchSchema.init_changeset(season_id)
        |> Repo.insert()
        |> parse_repo_response()
    end
  end

  # ----------------------------------------------------------------

  def join_match(match_id, user_id) do
    case Repo.get_by(MatchSchema, id: match_id) do
      nil ->
        {:error, :not_found}

      match ->
        match
        |> MatchSchema.join_changeset(user_id)
        |> Repo.update()
        |> parse_repo_response()
    end
  end

  def assign_color(match_id, user_id, color) do
    case Repo.get_by(MatchSchema, id: match_id) do
      nil ->
        {:error, :not_found}

      %{host_id: host_id, player_id: player_id} when user_id not in [host_id, player_id] ->
        {:error, :forbbiden}

      match ->
        color = if user_id == match.host_id, do: color, else: opposite_color(color)

        match
        |> MatchSchema.assign_color_changeset(color)
        |> Repo.update()
        |> parse_repo_response()
    end
  end

  def delete_match(match_id, user_id) do
    case Repo.get_by(MatchSchema, id: match_id) do
      nil ->
        {:error, :not_found}

      %{host_id: id} when id != user_id ->
        {:error, :forbbiden}

      match ->
        Repo.delete!(match)
        :ok
    end
  end

  # ----------------------------------------------------------------
  defp parse_repo_response({:ok, %{match: match}}), do: {:ok, MatchStruct.build_from_schema(match)}
  defp parse_repo_response({:ok, match}), do: {:ok, MatchStruct.build_from_schema(match)}
  defp parse_repo_response({:error, error}), do: {:error, error}
end
