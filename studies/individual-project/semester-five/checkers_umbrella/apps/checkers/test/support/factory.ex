defmodule Checkers.Factory do
  @moduledoc false
  use ExMachina.Ecto, repo: StaffWorkingHours.Repo
  alias Checkers.Schemas

  def user_factory do
    %Schemas.User{
      email: "example@example.com",
      password_hash: "random-hash"
    }
  end

  def match_factory do
    %Schemas.Match{
      id: Ecto.UUID.generate(),
      host_id: sequence("host_id", &(&1 + 1)),
      status: :initialized
    }
  end
end
