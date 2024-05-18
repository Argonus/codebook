defmodule Checkers.Matches.Impl do
  @moduledoc false
  @behaviour Checkers.Matches.Behaviour

  alias Checkers.Matches.MatchManagement

  @impl true
  defdelegate create_match(host_id), to: MatchManagement
end
