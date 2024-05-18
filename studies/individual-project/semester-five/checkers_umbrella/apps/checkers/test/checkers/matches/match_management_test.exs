defmodule Checkers.Matches.MatchManagementTest do
  use Checkers.DataCase, async: true

  use Hammox.Protect,
    module: Checkers.Matches.MatchManagement,
    behaviour: Checkers.Matches.Behaviour,
    functions: [:create_match]

  describe "create_match/1" do
    test "creates a match" do
      {:ok, match} = create_match(1)

      assert match.id
      assert match.host_id == 1
      assert match.status == :initialized
    end
  end
end
