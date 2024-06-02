defmodule Checkers.Matches.MatchStructTest do
  use ExUnit.Case, async: true
  import Checkers.Factory

  alias Checkers.Matches.MatchStruct

  describe "build_from_schema/2" do
    test "builds match struct from match schema without users" do
      match = build(:match)

      expected = %MatchStruct{
        id: match.id,
        host_id: match.host_id,
        player_id: match.player_id,
        status: match.status,
        board: default_board()
      }

      assert expected == MatchStruct.build_from_schema(match)
    end

    test "builds match struct from match schema with users" do
      host = build(:user)
      player = build(:user)
      match = build(:match, host: host, player: player)

      expected = %MatchStruct{
        id: match.id,
        host_id: match.host_id,
        host_name: match.host.login,
        host_color: nil,
        player_id: match.player_id,
        player_name: match.player.login,
        player_color: nil,
        status: match.status,
        board: default_board()
      }

      assert expected == MatchStruct.build_from_schema(match)
    end

    test "builds match schema with season number" do
      season = build(:season, season_number: 1)
      match = build(:match, season: season)

      expected = %MatchStruct{
        id: match.id,
        season_number: season.season_number,
        host_id: match.host_id,
        player_id: match.player_id,
        status: match.status,
        board: default_board()
      }

      assert expected == MatchStruct.build_from_schema(match)
    end
  end

  defp default_board, do: Checkers.Matches.Helpers.MatchHelpers.draw_initial_board()
end
