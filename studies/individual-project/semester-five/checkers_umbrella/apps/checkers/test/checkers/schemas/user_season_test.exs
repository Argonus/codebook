defmodule Checkers.Schemas.UserSeasonTest do
  use Checkers.DataCase, async: true
  import Checkers.Factory

  alias Checkers.Repo
  alias Checkers.Schemas.UserSeason

  describe "init_changeset/1" do
    setup do
      valid_params = %{
        user_id: insert(:user).id,
        season_id: insert(:season).id
      }

      {:ok, %{params: valid_params}}
    end

    test "create valid changeset", %{params: params} do
      changeset = UserSeason.init_changeset(params)

      assert changeset.valid?
    end

    test "allows to save changeset in db", %{params: params} do
      changeset = UserSeason.init_changeset(params)
      user_season = Repo.insert!(changeset)

      assert user_season.user_id == params.user_id
      assert user_season.season_id == params.season_id
      assert user_season.rating == 0
    end

    for field <- ~w(user_id season_id)a do
      test "returns error if #{field} is missing", %{params: params} do
        invalid_params = Map.put(params, unquote(field), nil)
        changeset = UserSeason.init_changeset(invalid_params)

        refute changeset.valid?
        assert {"can't be blank", [validation: :required]} = changeset.errors[unquote(field)]
      end
    end

    test "returns error if user relationship does not exists", %{params: params} do
      invalid_params = Map.put(params, :user_id, params.user_id + 1)
      changeset = UserSeason.init_changeset(invalid_params)

      {:error, changeset} = Repo.insert(changeset)

      assert {"does not exist", [{:constraint, :foreign}, {:constraint_name, "user_seasons_user_id_fkey"}]} =
               changeset.errors[:user_id]
    end

    test "returns error if season relationship does not exists", %{params: params} do
      invalid_params = Map.put(params, :season_id, Ecto.UUID.generate())
      changeset = UserSeason.init_changeset(invalid_params)

      {:error, changeset} = Repo.insert(changeset)

      assert {"does not exist", [{:constraint, :foreign}, {:constraint_name, "user_seasons_season_id_fkey"}]} =
               changeset.errors[:season_id]
    end
  end
end
