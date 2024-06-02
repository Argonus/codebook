defmodule CheckersWeb.MatchLive.FormComponent do
  use CheckersWeb, :live_component
  alias Checkers.Matches

  @impl true
  def render(assigns) do
    ~H"""
    <div>
      <.header>
        <%= @title %>
        <:subtitle>Use this form to manage match records in your database.</:subtitle>
      </.header>

      <.simple_form
        for={@form}
        id="match-form"
        phx-target={@myself}
        phx-change="validate"
        phx-submit="save"
      >
        <.input field={@form[:id]} type="text" label="Id" />
        <:actions>
          <.button phx-disable-with="Saving...">Save Match</.button>
        </:actions>
      </.simple_form>
    </div>
    """
  end

  @impl true
  def update(%{match: match} = assigns, socket) do
  end

  @impl true
  def handle_event("validate", %{"match" => match_params}, socket) do
  end

  def handle_event("save", %{"match" => match_params}, socket) do
  end

  defp save_match(socket, :edit, match_params) do
  end

  defp assign_form(socket, %Ecto.Changeset{} = changeset) do
  end

  defp notify_parent(msg), do: send(self(), {__MODULE__, msg})
end
