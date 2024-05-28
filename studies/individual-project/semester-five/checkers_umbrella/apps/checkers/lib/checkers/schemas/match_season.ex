defmodule Checkers.Schemas.MatchSeason do
  @moduledoc """
  Schema representing match seasons table
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  schema "match_seasons" do
    field :match_id, :binary_id, primary_key: true
    field :season_id, :binary_id, primary_key: true

    timestamps()
  end

  @required_params ~w(match_id season_id)a

  @type params :: %{
          user_id: String.t(),
          season_id: String.t()
        }

  @spec changeset(params) :: Ecto.Changeset.t()
  def changeset(params) do
    %__MODULE__{}
    |> cast(params, @required_params)
    |> validate_required(@required_params)
    |> foreign_key_constraint(:match_id)
    |> foreign_key_constraint(:season_id)
  end
end
