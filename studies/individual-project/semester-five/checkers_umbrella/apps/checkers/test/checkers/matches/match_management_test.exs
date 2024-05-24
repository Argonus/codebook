defmodule Checkers.Matches.MatchManagementTest do
  use Checkers.DataCase, async: true
  import Checkers.Factory

  alias Checkers.Matches.MatchStruct

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

  describe "create_match/1" do
    test "creates a match" do
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
end
