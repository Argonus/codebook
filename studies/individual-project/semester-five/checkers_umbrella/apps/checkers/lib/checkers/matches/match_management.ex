defmodule Checkers.Matches.MatchManagement do
  @moduledoc false

  alias Checkers.Matches.MatchStruct
  alias Checkers.Repo
  alias Checkers.Schemas.Match, as: MatchSchema

  def get_match(match_id) do
    case Repo.get_by(MatchSchema, id: match_id) do
      nil -> {:error, :not_found}
      match -> {:ok, MatchStruct.build_from_schema(match)}
    end
  end

  def create_match(host_id) do
    host_id
    |> MatchSchema.init_changeset()
    |> Repo.insert()
    |> parse_repo_response()
  end

  def join_match(match_id, user_id) do
    case Repo.get_by(MatchSchema, id: match_id) do
      nil ->
        {:error, :not_found}

      match ->
        match
        |> MatchSchema.join_changeset(user_id)
        |> Repo.update()
        |> parse_repo_response()
    end
  end

  # ----------------------------------------------------------------
  defp parse_repo_response({:ok, match}), do: {:ok, MatchStruct.build_from_schema(match)}
  defp parse_repo_response({:error, error}), do: {:error, error}
end
