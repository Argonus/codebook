defmodule Pooly.WorkerSupervisor do
  @moduledoc false

  use Supervisor

  #########
  #  API  #
  #########

  def start_link(pool_server, mfa) do
    Supervisor.start_link(__MODULE__, [pool_server, mfa])
  end

  #############
  # Callbacks #
  #############

  def init([pool_server, {m, f, a} = _mfa]) do
    Process.link(pool_server)

    worker_opts = [
      restart: :temporary,
      shutdown: 5000,
      function: f
    ]

    children = [worker(m, a, worker_opts)]
    opts = [strategy: :simple_one_for_one, max_restarts: 5, max_seconds: 5]

    Supervisor.init(children, opts)
  end
end
