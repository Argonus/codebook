defmodule Checkers.Matches.MatchStructTest do
  use ExUnit.Case, async: true
  import Checkers.Factory

  alias Checkers.Matches.MatchStruct

  describe "build_from_schema/2" do
    test "builds match struct from match schema" do
      match = build(:match)

      assert %MatchStruct{
               id: match.id,
               host_id: match.host_id,
               player_id: match.player_id,
               status: match.status
             } == MatchStruct.build_from_schema(match)
    end
  end
end
