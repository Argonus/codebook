defmodule Checkers.Repo.Migrations.CreateMatchesTable do
  use Ecto.Migration

  def up do
    execute "CREATE TYPE match_status AS ENUM ('initialized', 'pending', 'completed')"
    execute "CREATE EXTENSION IF NOT EXISTS pgcrypto"

    create table(:matches, primary_key: false) do
      add :id, :uuid, primary_key: true, default: fragment("gen_random_uuid()")
      # Relationships
      add :host_id, :bigint, null: false
      add :player_id, :bigint
      add :winner_id, :bigint
      # Game State
      add :status, :match_status, null: false
      add :moves, :map, null: false

      timestamps()
    end

    create index(:matches, [:host_id])
    create index(:matches, [:player_id])
  end

  def down do
    drop table(:matches)
    execute "DROP TYPE match_status"
  end
end
