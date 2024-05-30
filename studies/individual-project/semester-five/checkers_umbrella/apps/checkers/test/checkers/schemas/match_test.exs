defmodule Checkers.Schemas.MatchTest do
  use Checkers.DataCase, async: true
  import Checkers.Factory

  alias Checkers.Repo
  alias Checkers.Schemas.Match

  describe "init_changeset/1" do
    setup do
      season = insert(:season)
      {:ok, season_id: season.id}
    end

    test "returns a changeset with the correct attributes", %{season_id: season_id} do
      changeset = Match.init_changeset(1, season_id)

      assert changeset.valid?
    end

    test "creates match with valid attributes", %{season_id: season_id} do
      changeset = Match.init_changeset(2, season_id)
      match = Repo.insert!(changeset)

      assert match.host_id == 2
      assert match.status == :initialized
      assert match.moves == %{}
    end

    test "return error when host_id is nil", %{season_id: season_id} do
      changeset = Match.init_changeset(nil, season_id)

      {:error, changeset} = Repo.insert(changeset)
      assert {"can't be blank", [validation: :required]} = changeset.errors[:host_id]
    end

    test "return error when season_id is nil" do
      changeset = Match.init_changeset(1, nil)

      {:error, changeset} = Repo.insert(changeset)
      assert {"can't be blank", [validation: :required]} = changeset.errors[:season_id]
    end
  end

  describe "join_changeset/2" do
    test "creates valid changeset" do
      match = insert(:match)
      changeset = Match.join_changeset(match, 2)

      assert changeset.valid?
    end

    test "updates match with valid attributes" do
      match = insert(:match)
      changeset = Match.join_changeset(match, 2)
      match = Repo.update!(changeset)

      assert match.player_id == 2
    end

    test "return error when host_id is nil" do
      match = insert(:match)
      changeset = Match.join_changeset(match, nil)

      {:error, changeset} = Repo.update(changeset)
      assert {"can't be blank", [validation: :required]} = changeset.errors[:player_id]
    end
  end

  describe "assign_colors_changeset/2" do
    test "assigns valid color to host" do
      match = insert(:match)
      changeset = Match.assign_color_changeset(match, :black)

      assert changeset.valid?
    end

    test "updates match with valid attributes" do
      match = insert(:match)
      changeset = Match.assign_color_changeset(match, :black)
      match = Repo.update!(changeset)

      assert match.host_color == :black
    end

    test "return error when color is invalid" do
      match = insert(:match)
      changeset = Match.assign_color_changeset(match, :blue)

      {:error, changeset} = Repo.update(changeset)
      assert {"is invalid", _} = changeset.errors[:host_color]
    end
  end
end
