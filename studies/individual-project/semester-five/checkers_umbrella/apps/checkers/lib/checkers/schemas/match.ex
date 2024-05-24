defmodule Checkers.Schemas.Match do
  @moduledoc """
  Represents a match stored in the database.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{}

  @primary_key {:id, :binary_id, autogenerate: true}
  schema "matches" do
    # Relationships
    field :host_id, :integer
    field :player_id, :integer
    field :winner_id, :integer
    # Game State
    field :status, Ecto.Enum, values: ~w(initialized pending completed)a
    field :moves, :map

    timestamps()
  end

  @required_attributes ~w(host_id status moves)a

  @doc """
  Initializes a changeset for a match.
  """
  @spec init_changeset(non_neg_integer()) :: Ecto.Changeset.t()
  def init_changeset(host_id) do
    %__MODULE__{}
    |> cast(%{host_id: host_id}, [:host_id])
    |> cast(%{status: :initialized}, [:status])
    |> cast(%{moves: %{}}, [:moves])
    |> validate_required(@required_attributes)
  end

  @doc """
  Assigns new user to match
  """
  @spec join_changeset(__MODULE__.t(), non_neg_integer) :: Ecto.Changeset.t()
  def join_changeset(match, user_id) do
    match
    |> cast(%{player_id: user_id}, [:player_id])
    |> validate_required([:player_id])
  end
end
