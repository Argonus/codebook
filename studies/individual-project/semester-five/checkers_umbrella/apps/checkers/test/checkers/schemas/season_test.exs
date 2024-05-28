defmodule Checkers.Schemas.SeasonTest do
  use Checkers.DataCase, async: true

  alias Checkers.Repo
  alias Checkers.Schemas.Season

  describe "init_changeset" do
    setup do
      valid_parms = %{
        start_datetime_utc: ~U[2023-01-01 00:00:00.000000Z],
        end_datetime_utc: ~U[2023-01-01 00:00:00.000000Z],
        season_number: 1
      }

      {:ok, %{params: valid_parms}}
    end

    test "create valid season changeset", %{params: params} do
      changeset = Season.init_changeset(params)

      assert changeset.valid?
    end

    test "allows to save changeset in db", %{params: params} do
      changeset = Season.init_changeset(params)

      season = Repo.insert!(changeset)

      assert season.season_number == params.season_number
      assert season.start_datetime_utc == params.start_datetime_utc
      assert season.end_datetime_utc == params.end_datetime_utc
    end

    for field <- ~w(season_number start_datetime_utc end_datetime_utc)a do
      test "returns error if #{field} is missing", %{params: params} do
        invalid_params = Map.put(params, unquote(field), nil)

        changeset = Season.init_changeset(invalid_params)

        refute changeset.valid?
        assert {"can't be blank", [validation: :required]} = changeset.errors[unquote(field)]
      end
    end

    test "validates uniqueness of number", %{params: params} do
      changeset = Season.init_changeset(params)
      _ = Repo.insert!(changeset)

      {:error, changeset} = Repo.insert(changeset)

      assert {"has already been taken", [{:constraint, :unique}, {:constraint_name, "seasons_season_number_index"}]} =
               changeset.errors[:season_number]
    end
  end
end
