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

  def default_board do
    [
      [
        %{color: :black, pawn: nil, id: "0x0"},
        %{color: :white, pawn: nil, id: "0x1"},
        %{color: :black, pawn: nil, id: "0x2"},
        %{color: :white, pawn: nil, id: "0x3"},
        %{color: :black, pawn: nil, id: "0x4"},
        %{color: :white, pawn: nil, id: "0x5"},
        %{color: :black, pawn: nil, id: "0x6"},
        %{color: :white, pawn: nil, id: "0x7"}
      ],
      [
        %{color: :white, pawn: nil, id: "1x0"},
        %{color: :black, pawn: nil, id: "1x1"},
        %{color: :white, pawn: nil, id: "1x2"},
        %{color: :black, pawn: nil, id: "1x3"},
        %{color: :white, pawn: nil, id: "1x4"},
        %{color: :black, pawn: nil, id: "1x5"},
        %{color: :white, pawn: nil, id: "1x6"},
        %{color: :black, pawn: nil, id: "1x7"}
      ],
      [
        %{color: :black, pawn: nil, id: "2x0"},
        %{color: :white, pawn: nil, id: "2x1"},
        %{color: :black, pawn: nil, id: "2x2"},
        %{color: :white, pawn: nil, id: "2x3"},
        %{color: :black, pawn: nil, id: "2x4"},
        %{color: :white, pawn: nil, id: "2x5"},
        %{color: :black, pawn: nil, id: "2x6"},
        %{color: :white, pawn: nil, id: "2x7"}
      ],
      [
        %{color: :white, pawn: nil, id: "3x0"},
        %{color: :black, pawn: nil, id: "3x1"},
        %{color: :white, pawn: nil, id: "3x2"},
        %{color: :black, pawn: nil, id: "3x3"},
        %{color: :white, pawn: nil, id: "3x4"},
        %{color: :black, pawn: nil, id: "3x5"},
        %{color: :white, pawn: nil, id: "3x6"},
        %{color: :black, pawn: nil, id: "3x7"}
      ],
      [
        %{color: :black, pawn: nil, id: "4x0"},
        %{color: :white, pawn: nil, id: "4x1"},
        %{color: :black, pawn: nil, id: "4x2"},
        %{color: :white, pawn: nil, id: "4x3"},
        %{color: :black, pawn: nil, id: "4x4"},
        %{color: :white, pawn: nil, id: "4x5"},
        %{color: :black, pawn: nil, id: "4x6"},
        %{color: :white, pawn: nil, id: "4x7"}
      ],
      [
        %{color: :white, pawn: nil, id: "5x0"},
        %{color: :black, pawn: nil, id: "5x1"},
        %{color: :white, pawn: nil, id: "5x2"},
        %{color: :black, pawn: nil, id: "5x3"},
        %{color: :white, pawn: nil, id: "5x4"},
        %{color: :black, pawn: nil, id: "5x5"},
        %{color: :white, pawn: nil, id: "5x6"},
        %{color: :black, pawn: nil, id: "5x7"}
      ],
      [
        %{color: :black, pawn: nil, id: "6x0"},
        %{color: :white, pawn: nil, id: "6x1"},
        %{color: :black, pawn: nil, id: "6x2"},
        %{color: :white, pawn: nil, id: "6x3"},
        %{color: :black, pawn: nil, id: "6x4"},
        %{color: :white, pawn: nil, id: "6x5"},
        %{color: :black, pawn: nil, id: "6x6"},
        %{color: :white, pawn: nil, id: "6x7"}
      ],
      [
        %{color: :white, pawn: nil, id: "7x0"},
        %{color: :black, pawn: nil, id: "7x1"},
        %{color: :white, pawn: nil, id: "7x2"},
        %{color: :black, pawn: nil, id: "7x3"},
        %{color: :white, pawn: nil, id: "7x4"},
        %{color: :black, pawn: nil, id: "7x5"},
        %{color: :white, pawn: nil, id: "7x6"},
        %{color: :black, pawn: nil, id: "7x7"}
      ]
    ]
  end
end
