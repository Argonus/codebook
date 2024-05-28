defmodule Checkers.Seasons.SeasonManagement do
  @moduledoc false

  alias Checkers.Repo
  alias Checkers.Schemas.Season, as: SeasonSchema
  alias Checkers.Seasons.SeasonStruct
  alias Checkers.Schemas.User, as: UserSchema
  alias Checkers.Schemas.UserSeason, as: UserSeasonSchema

  @season_duration 14

  def join_season(season_id, user_id) do
    case [get_season(season_id), get_user(user_id)] do
      [nil, _] ->
        {:error, :season_not_found}

      [_, nil] ->
        {:error, :user_not_found}

      _ ->
        %{season_id: season_id, user_id: user_id}
        |> UserSeasonSchema.init_changeset()
        |> Repo.insert()
        |> case do
          {:ok, _} -> :ok
          result -> result
        end
    end
  end

  def get_current_season do
    case do_get_current_season() do
      nil -> {:error, :not_found}
      season -> {:ok, SeasonStruct.build(season)}
    end
  end

  def init_season do
    season = get_prev_season()
    number = if season, do: season.season_number + 1, else: 1

    start_timestamp =
      if season do
        season.end_datetime_utc |> DateTime.add(1, :second)
      else
        DateTime.utc_now() |> Timex.beginning_of_day()
      end

    end_timestamp = start_timestamp |> DateTime.add(@season_duration, :day) |> Timex.end_of_day()

    %{
      season_number: number,
      start_datetime_utc: start_timestamp,
      end_datetime_utc: end_timestamp
    }
    |> SeasonSchema.init_changeset()
    |> Repo.insert()
    |> parse_repo_response()
  end

  def get_season(season_id) do
    case Repo.get(SeasonSchema, season_id) do
      nil -> nil
      season -> SeasonStruct.build(season)
    end
  end

  def get_user(user_id) do
    case Repo.get(UserSchema, user_id) do
      nil -> nil
      user -> user
    end
  end

  # ----------------------------------------------------------------
  defp do_get_current_season do
    import Ecto.Query

    Repo.one(
      from s in SeasonSchema,
        where: s.end_datetime_utc > ^DateTime.utc_now() and s.start_datetime_utc <= ^DateTime.utc_now(),
        order_by: {:desc, s.end_datetime_utc},
        limit: 1
    )
  end

  # ----------------------------------------------------------------
  defp get_prev_season do
    import Ecto.Query

    Repo.one(from s in SeasonSchema, order_by: {:desc, s.end_datetime_utc}, limit: 1)
  end

  defp parse_repo_response({:ok, schema}), do: {:ok, SeasonStruct.build(schema)}
  defp parse_repo_response(result), do: result
end
