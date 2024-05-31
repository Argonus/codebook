defmodule Checkers.Schemas.Match do
  @moduledoc """
  Represents a match stored in the database.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{}
  @type color :: :black | :white

  @primary_key {:id, :binary_id, autogenerate: true}
  schema "matches" do
    # Relationships
    belongs_to :host, Checkers.Schemas.User
    belongs_to :player, Checkers.Schemas.User
    belongs_to :season, Checkers.Schemas.Season, foreign_key: :season_id, references: :id, type: :binary_id

    field :winner_id, :integer
    # Game State
    field :status, Ecto.Enum, values: ~w(initialized pending completed)a
    field :host_color, Ecto.Enum, values: ~w(black white)a
    field :moves, :map

    timestamps()
  end

  @required_attributes ~w(host_id season_id status moves)a
  @valid_colors ~w(black white)a

  @doc """
  Initializes a changeset for a match.
  """
  @spec init_changeset(non_neg_integer(), String.t()) :: Ecto.Changeset.t()
  def init_changeset(host_id, season_id) do
    %__MODULE__{}
    |> cast(%{host_id: host_id, season_id: season_id}, [:host_id, :season_id])
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
    |> validate_change(:player_id, fn :player_id, new_val ->
      cond do
        new_val == match.host_id -> [player_id: "Player cannot be the same as host"]
        !is_nil(match.player_id) -> [player_id: "Player already assigned"]
        true -> []
      end
    end)
    |> validate_required([:player_id])
  end

  @spec assign_color_changeset(__MODULE__.t(), color) :: Ecto.Changeset.t()
  def assign_color_changeset(match, host_color) do
    match
    |> cast(%{host_color: host_color}, [:host_color])
    |> cast(%{status: :pending}, [:status])
    |> validate_inclusion(:host_color, @valid_colors)
  end
end
