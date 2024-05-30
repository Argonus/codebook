defmodule Checkers.Repo.Migrations.CreateLeaderboardTables do
  use Ecto.Migration

  def change do
    create table(:seasons, primary_key: false) do
      add :id, :uuid, primary_key: true, default: fragment("gen_random_uuid()")
      add :season_number, :integer
      add :start_datetime_utc, :utc_datetime_usec
      add :end_datetime_utc, :utc_datetime_usec

      timestamps()
    end

    create index(:seasons, [:start_datetime_utc, "end_datetime_utc DESC"])
    create index(:seasons, :season_number, unique: true)

    alter table(:matches) do
      add :season_id, references(:seasons, type: :uuid)
    end

    create index(:matches, :season_id)

    # References as this is required
    create table(:user_seasons, primary_key: false) do
      add :user_id, references(:users), primary_key: true
      add :season_id, references(:seasons, type: :uuid), primary_key: true
      add :rating, :bigint, null: false, default: 0

      timestamps()
    end

    create index(:user_seasons, :rating)
  end
end
