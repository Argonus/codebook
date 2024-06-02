defmodule CheckersWeb.MatchLive.Show do
  use CheckersWeb, :live_view
  use CheckersWeb.AuthHelper, otp_app: :checkers_web
  alias Checkers.Matches

  @impl true
  def mount(_params, session, socket) do
    {:ok, current_user(socket, session)}
  end

  @impl true
  def handle_params(%{"id" => id}, _session, socket) do
    {:ok, match} = Matches.get_match(id)

    {:noreply,
     socket
     |> assign(:page_title, page_title(socket.assigns.live_action))
     |> assign(:match, match)}
  end

  defp page_title(:show), do: "Show Match"
  defp page_title(:edit), do: "Edit Match"
end
