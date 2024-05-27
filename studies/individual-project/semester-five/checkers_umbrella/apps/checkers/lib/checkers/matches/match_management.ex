defmodule Checkers.Matches.MatchManagement do
  @moduledoc false
  import Checkers.Matches.Helpers.MatchHelpers

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

  def assign_color(match_id, user_id, color) do
    case Repo.get_by(MatchSchema, id: match_id) do
      nil ->
        {:error, :not_found}

      %{host_id: host_id, player_id: player_id} when user_id not in [host_id, player_id] ->
        {:error, :forbbiden}

      match ->
        color = if user_id == match.host_id, do: color, else: opposite_color(color)

        match
        |> MatchSchema.assign_color_changeset(color)
        |> Repo.update()
        |> parse_repo_response()
    end
  end

  def delete_match(match_id, user_id) do
    case Repo.get_by(MatchSchema, id: match_id) do
      nil ->
        {:error, :not_found}

      %{host_id: id} when id != user_id ->
        {:error, :forbbiden}

      match ->
        Repo.delete!(match)
        :ok
    end
  end

  # ----------------------------------------------------------------
  defp parse_repo_response({:ok, match}), do: {:ok, MatchStruct.build_from_schema(match)}
  defp parse_repo_response({:error, error}), do: {:error, error}
end
