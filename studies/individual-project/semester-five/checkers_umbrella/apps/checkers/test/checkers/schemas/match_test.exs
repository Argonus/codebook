defmodule Checkers.Schemas.MatchTest do
  use Checkers.DataCase, async: true
  import Checkers.Factory

  alias Checkers.Repo
  alias Checkers.Schemas.Match

  describe "init_changeset/1" do
    test "returns a changeset with the correct attributes" do
      changeset = Match.init_changeset(1)

      assert changeset.valid?
    end

    test "creates match with valid attributes" do
      changeset = Match.init_changeset(2)
      match = Repo.insert!(changeset)

      assert match.host_id == 2
      assert match.status == :initialized
      assert match.moves == %{}
    end

    test "return error when host_id is nil" do
      changeset = Match.init_changeset(nil)

      {:error, changeset} = Repo.insert(changeset)
      assert {"can't be blank", [validation: :required]} = changeset.errors[:host_id]
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
end
