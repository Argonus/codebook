defmodule Checkers.Repo.Migrations.AddColorsToMatch do
  use Ecto.Migration

  def up do
    execute "CREATE TYPE player_color AS ENUM ('white', 'black')"

    alter table(:matches) do
      add :host_color, :player_color
    end
  end

  def down do
    alter table(:matches) do
      remove :host_color
    end

    execute "DROP TYPE player_color AS ENUM ('white', 'black')"
  end
end
