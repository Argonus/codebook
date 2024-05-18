defmodule Checkers.Matches.MatchStruct do
  @moduledoc """
  This module is a struct for the Match.
  """
  use TypedStruct

  typedstruct do
    @typedoc "A match"

    field(:id, String.t())
    field(:host_id, integer())
    field(:status, :initialized)
  end
end
