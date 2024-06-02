defmodule Checkers.Schemas.Season do
  @moduledoc """
  Season schema
  - Each season has start datetime & end datetime
  - Each season has different leaderboard
  """
  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{}

  @primary_key {:id, :binary_id, autogenerate: true}
  schema "seasons" do
    field :season_number, :integer
    field :start_datetime_utc, :utc_datetime_usec
    field :end_datetime_utc, :utc_datetime_usec

    timestamps()
  end

  @params_required ~w(season_number start_datetime_utc end_datetime_utc)a

  @type params :: %{
          season_number: pos_integer,
          start_datetime_utc: DateTime.t(),
          end_datetime_utc: DateTime.t()
        }

  @doc """
  Changeset used to initialize season
  """
  @spec init_changeset(params) :: Ecto.Changeset.t()
  def init_changeset(params) do
    %__MODULE__{}
    |> cast(params, @params_required)
    |> validate_required(@params_required)
    |> unique_constraint(:season_number)
  end
end
