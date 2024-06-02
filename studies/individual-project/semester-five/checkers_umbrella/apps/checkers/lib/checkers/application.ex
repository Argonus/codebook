defmodule Checkers.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      Checkers.Repo,
      {DNSCluster, query: Application.get_env(:checkers, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: Checkers.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: Checkers.Finch}
      # Start a worker by calling: Checkers.Worker.start_link(arg)
      # {Checkers.Worker, arg}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: Checkers.Supervisor)
  end
end
