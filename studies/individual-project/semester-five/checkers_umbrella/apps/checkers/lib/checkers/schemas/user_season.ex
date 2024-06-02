defmodule Checkers.Schemas.UserSeason do
  @moduledoc """
  Relationship table between user & season
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  schema "user_seasons" do
    field :user_id, :integer, primary_key: true
    field :season_id, :binary_id, primary_key: true
    field :rating, :integer, default: 0

    timestamps()
  end

  @type params :: %{
          user_id: pos_integer,
          season_id: String.t()
        }
  @spec init_changeset(params) :: Ecto.Changeset.t()
  def init_changeset(params) do
    %__MODULE__{}
    |> cast(params, [:user_id, :season_id])
    |> validate_required([:user_id, :season_id])
    |> foreign_key_constraint(:user_id)
    |> foreign_key_constraint(:season_id)
  end
end
