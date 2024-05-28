defmodule Checkers.Seasons.SeasonStructTest do
  use ExUnit.Case, async: true
  import Checkers.Factory

  alias Checkers.Seasons.SeasonStruct

  describe "build/1" do
    test "builds active season struct" do
      season = build(:season, id: Ecto.UUID.generate())

      result = SeasonStruct.build(season)

      assert result.season_id == season.id
      assert result.start_datetime == season.start_datetime_utc
      assert result.end_datetime == season.end_datetime_utc
      assert result.active
    end
  end
end
