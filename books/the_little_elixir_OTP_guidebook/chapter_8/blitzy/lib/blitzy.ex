defmodule Blitzy do
  @moduledoc false

  use Application

  def start(_type, _args) do
    Blitzy.Supervisor.start_link(:ok)
  end
end
