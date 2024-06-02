defmodule Checkers.Seasons.SeasonManagementTest do
  use Checkers.DataCase, async: true
  import Checkers.Factory

  alias Checkers.Repo
  alias Checkers.Schemas.UserSeason

  describe "init_season/0" do
    test "creates a new season" do
      {:ok, season} = Checkers.Seasons.Impl.init_season()

      assert season.season_number
      assert season.start_datetime
      assert season.end_datetime
      assert season.active
    end

    test "creates a season based on the previous one" do
      insert(:season,
        start_datetime_utc: ~U[2020-01-01 23:59:59.000000Z],
        end_datetime_utc: ~U[2020-01-15 23:59:59.000000Z]
      )

      pre_season =
        insert(:season,
          start_datetime_utc: ~U[2020-01-16 00:00:00.000000Z],
          end_datetime_utc: ~U[2020-01-30 23:59:59.000000Z]
        )

      {:ok, season} = Checkers.Seasons.Impl.init_season()

      assert season.season_number == pre_season.season_number + 1
      assert season.start_datetime == ~U[2020-01-31 00:00:00.000000Z]
      assert season.end_datetime == ~U[2020-02-14 23:59:59.999999Z]
    end
  end

  describe "get_current_season/0" do
    test "returns the current season" do
      {:ok, new_season} = Checkers.Seasons.Impl.init_season()

      {:ok, season} = Checkers.Seasons.Impl.get_current_season()

      assert season.season_number == new_season.season_number
      assert season.start_datetime == new_season.start_datetime
      assert season.end_datetime == new_season.end_datetime
      assert season.active
    end

    test "returns most recent season" do
      start_timestamp = DateTime.utc_now() |> DateTime.add(-28, :day)
      end_timestamp = start_timestamp |> DateTime.add(14, :day)

      s1 = insert(:season, start_datetime_utc: start_timestamp, end_datetime_utc: end_timestamp)

      {:ok, new_season} = Checkers.Seasons.Impl.init_season()
      assert new_season.season_number == s1.season_number + 1

      {:ok, season} = Checkers.Seasons.Impl.get_current_season()
      assert season.season_number == new_season.season_number
    end

    test "returns :not_found if there is no current season" do
      assert {:error, :not_found} = Checkers.Seasons.Impl.get_current_season()
    end
  end

  describe "join_season/2" do
    test "joins a user to a season" do
      season = insert(:season)
      user = insert(:user)

      :ok = Checkers.Seasons.Impl.join_season(season.id, user.id)

      assert Repo.get_by(UserSeason, season_id: season.id, user_id: user.id)
    end

    test "returns :season_not_found if the season does not exist" do
      user = insert(:user)

      assert {:error, :season_not_found} = Checkers.Seasons.Impl.join_season(Ecto.UUID.generate(), user.id)
    end

    test "returns :user_not_found if the user does not exist" do
      season = insert(:season)

      assert {:error, :user_not_found} = Checkers.Seasons.Impl.join_season(season.id, 123)
    end
  end
end
