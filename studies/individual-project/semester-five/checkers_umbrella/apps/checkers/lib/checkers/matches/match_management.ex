defmodule Checkers.Matches.MatchManagement do
  @moduledoc false

  alias Checkers.Matches.MatchStruct
  alias Checkers.Repo
  alias Checkers.Schemas.Match, as: MatchSchema

  def create_match(host_id) do
    host_id
    |> MatchSchema.init_changeset()
    |> Repo.insert()
    |> case do
      {:ok, match} -> {:ok, %MatchStruct{id: match.id, host_id: match.host_id, status: match.status}}
      {:error, changeset} -> {:error, changeset}
    end
  end
end
