defmodule Checkers.Seasons.SeasonStruct do
  @moduledoc """
  Struct representing a season
  """
  use TypedStruct
  alias Checkers.Schemas.Season

  typedstruct do
    field :season_id, String.t(), enforce: true
    field :season_number, pos_integer, enforce: true
    field :start_datetime, DateTime.t(), enforce: true
    field :end_datetime, DateTime.t(), enfroce: true
    field :active, boolean, enfroce: true
  end

  @spec build(Season.t()) :: __MODULE__.t()
  def build(season_schema) do
    %__MODULE__{
      season_id: season_schema.id,
      season_number: season_schema.season_number,
      start_datetime: season_schema.start_datetime_utc,
      end_datetime: season_schema.end_datetime_utc,
      active: is_active(season_schema)
    }
  end

  defp is_active(%{start_datetime_utc: start_datetime, end_datetime_utc: end_datetime}) do
    current_timestamp = DateTime.utc_now()

    Timex.between?(current_timestamp, start_datetime, end_datetime)
  end
end
