defmodule CheckersWeb.MatchController do
  @moduledoc false
  use CheckersWeb, :controller
  plug :require_authenticated_user

  alias CheckersWeb.Router.Helpers, as: Routes
  alias Checkers.Matches

  def create(conn, _attrs) do
    current_user = Pow.Plug.current_user(conn)

    case Matches.create_match(current_user.id) do
      {:ok, _match} ->
        conn
        |> put_flash(:info, "Match created")
        |> redirect(to: Routes.page_path(conn, :home))

      {:error, _reason} ->
        conn
        |> put_flash(:error, "Failed to create match")
        |> redirect(to: Routes.page_path(conn, :home))
    end
  end

  def join(conn, params) do
    current_user = Pow.Plug.current_user(conn)
    match_id = Map.get(params, "match_id")

    case Matches.join_match(match_id, current_user.id) do
      {:ok, match} ->
        conn
        |> put_flash(:info, "Joined to match")
        |> redirect(to: Routes.match_show_path(conn, :show, match.id))

      {:error, :not_found} ->
        conn
        |> put_flash(:error, "Match not found")
        |> redirect(to: Routes.page_path(conn, :home))

      {:error, changeset = %Ecto.Changeset{}} ->
        error = fetch_changeset_errors(changeset)

        conn
        |> put_flash(:error, error)
        |> redirect(to: Routes.page_path(conn, :home))
    end
  end

  defp fetch_changeset_errors(changeset) do
    changeset.errors |> hd |> elem(1) |> elem(0)
  end

  def delete(conn, params) do
    current_user = Pow.Plug.current_user(conn)
    match_id = Map.get(params, "id")

    case Matches.delete_match(match_id, current_user.id) do
      :ok ->
        conn
        |> put_flash(:info, "Match deleted")
        |> redirect(to: Routes.page_path(conn, :home))

      {:error, :not_found} ->
        conn
        |> put_flash(:info, "Match not found")
        |> redirect(to: Routes.page_path(conn, :home))

      {:error, :forbbiden} ->
        conn
        |> put_flash(:error, "Action not allowed")
        |> redirect(to: Routes.page_path(conn, :home))
    end
  end

  # Private
  defp require_authenticated_user(conn, _opts) do
    if Pow.Plug.current_user(conn) do
      conn
    else
      conn
      |> put_flash(:error, "You must be logged in to access this page.")
      |> redirect(to: Routes.pow_session_path(conn, :new))
      |> halt()
    end
  end
end
