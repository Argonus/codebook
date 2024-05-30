defmodule Checkers.Matches.MatchManagementTest do
  use Checkers.DataCase, async: true
  import Checkers.Factory
  import Hammox

  alias Checkers.Matches.MatchStruct
  alias Checkers.Repo
  alias Checkers.Schemas.Match, as: MatchSchema

  use Hammox.Protect,
    module: Checkers.Matches.MatchManagement,
    behaviour: Checkers.Matches.Behaviour

  describe "get_match/1" do
    test "returns match if exists" do
      match = insert(:match)

      {:ok, result} = get_match(match.id)

      assert result == MatchStruct.build_from_schema(match)
    end

    test "returns error when match not found" do
      {:error, error_code} = get_match(Ecto.UUID.generate())

      assert error_code == :not_found
    end
  end

  describe "get_season_matches/1" do
    setup do
      season = insert(:season)

      {:ok, season: season}
    end

    test "returns all matches for a season", %{season: season} do
      match = insert(:match, season_id: season.id)
      [result] = get_season_matches(season.id)

      assert result == MatchStruct.build_from_schema(match)
    end

    test "returns empty list when no matches for a season", %{season: season} do
      assert get_season_matches(season.id) == []
    end

    test "filters matches by season", %{season: season} do
      other_season = insert(:season)
      _match = insert(:match, season_id: other_season.id)
      assert get_season_matches(season.id) == []
    end
  end

  describe "create_match/1" do
    setup do
      season = insert(:season)

      {:ok, season: season}
    end

    test "creates a match", %{season: season} do
      expect(Checkers.SeasonsMock, :get_current_season, fn ->
        {:ok, Checkers.Seasons.SeasonStruct.build(season)}
      end)

      {:ok, match} = create_match(1)

      assert match.id
      assert match.host_id == 1
      assert match.status == :initialized
    end
  end

  describe "join_match/2" do
    test "assigns new user to match" do
      match = insert(:match)
      {:ok, updated_match} = join_match(match.id, 2)

      assert updated_match.player_id == 2
    end

    test "does not change other match data" do
      match = insert(:match)
      {:ok, updated_match} = join_match(match.id, 2)

      assert updated_match.id == match.id
      assert updated_match.host_id == match.host_id
      assert updated_match.status == :initialized
    end

    test "returns error when match not found" do
      {:error, error_code} = join_match(Ecto.UUID.generate(), 2)

      assert error_code == :not_found
    end
  end

  describe "delete_match/2" do
    test "deletes existing match" do
      match = insert(:match)

      :ok = delete_match(match.id, match.host_id)

      refute Repo.get_by(MatchSchema, id: match.id)
    end

    test "returns error when player is not host" do
      match = insert(:match)

      {:error, error_code} = delete_match(match.id, match.host_id + 1)

      assert error_code == :forbbiden
      assert Repo.get_by(MatchSchema, id: match.id)
    end

    test "returns error when match not found" do
      {:error, error_code} = delete_match(Ecto.UUID.generate(), 1)

      assert error_code == :not_found
    end
  end

  describe "assign_color/3" do
    test "test assigns selected color if user is host" do
      match = insert(:match, host_id: 123)

      {:ok, match} = assign_color(match.id, match.host_id, :black)

      assert match.host_color == :black
      assert match.player_color == :white
    end

    test "test assigns opposite color if user is host" do
      match = insert(:match, host_id: 123, player_id: 321)

      {:ok, match} = assign_color(match.id, match.player_id, :white)

      assert match.host_color == :black
      assert match.player_color == :white
    end

    test "moves match to pending status" do
      match = insert(:match, host_id: 123)

      {:ok, match} = assign_color(match.id, match.host_id, :black)

      assert match.status == :pending
    end

    test "returns error when match is not found" do
      match = insert(:match, host_id: 123)

      {:error, error_code} = assign_color(Ecto.UUID.generate(), match.host_id, :black)

      assert error_code == :not_found
    end

    test "returns error when match does not belong to host" do
      match = insert(:match, host_id: 123, player_id: 321)

      {:error, error_code} = assign_color(match.id, 111, :black)

      assert error_code == :forbbiden
    end
  end
end
