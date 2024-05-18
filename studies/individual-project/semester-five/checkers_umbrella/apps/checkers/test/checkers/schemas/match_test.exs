defmodule Checkers.Schemas.MatchTest do
  use Checkers.DataCase, async: true

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
end
