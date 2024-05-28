defmodule Checkers.Schemas.MatchSeasonTest do
  use Checkers.DataCase, async: true
  import Checkers.Factory

  alias Checkers.Schemas.MatchSeason

  describe "changeset/1" do
    setup do
      valid_params = %{
        match_id: insert(:match).id,
        season_id: insert(:season).id
      }

      {:ok, %{params: valid_params}}
    end

    test "create valid changeset", %{params: params} do
      changeset = MatchSeason.changeset(params)

      assert changeset.valid?
    end

    test "allows to save changeset in db", %{params: params} do
      changeset = MatchSeason.changeset(params)
      user_season = Repo.insert!(changeset)

      assert user_season.match_id == params.match_id
      assert user_season.season_id == params.season_id
    end

    for field <- ~w(match_id season_id)a do
      test "returns error if #{field} is missing", %{params: params} do
        invalid_params = Map.put(params, unquote(field), nil)
        changeset = MatchSeason.changeset(invalid_params)

        refute changeset.valid?
        assert {"can't be blank", [validation: :required]} = changeset.errors[unquote(field)]
      end
    end

    test "returns error if match relationship does not exists", %{params: params} do
      invalid_params = Map.put(params, :match_id, Ecto.UUID.generate())
      changeset = MatchSeason.changeset(invalid_params)

      {:error, changeset} = Repo.insert(changeset)

      assert {"does not exist", [{:constraint, :foreign}, {:constraint_name, "match_seasons_match_id_fkey"}]} =
               changeset.errors[:match_id]
    end

    test "returns error if season relationship does not exists", %{params: params} do
      invalid_params = Map.put(params, :season_id, Ecto.UUID.generate())
      changeset = MatchSeason.changeset(invalid_params)

      {:error, changeset} = Repo.insert(changeset)

      assert {"does not exist", [{:constraint, :foreign}, {:constraint_name, "match_seasons_season_id_fkey"}]} =
               changeset.errors[:season_id]
    end
  end
end
