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
