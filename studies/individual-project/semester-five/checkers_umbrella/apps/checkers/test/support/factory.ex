defmodule Checkers.Factory do
  @moduledoc false
  use ExMachina.Ecto, repo: Checkers.Repo
  alias Checkers.Schemas

  def user_factory do
    %Schemas.User{
      login: "example",
      email: "example@example.com",
      password_hash: "random-hash"
    }
  end

  def match_factory do
    %Schemas.Match{
      id: Ecto.UUID.generate(),
      host_id: sequence("host_id", &(&1 + 1)),
      status: :initialized,
      moves: %{}
    }
  end

  def season_factory do
    %Schemas.Season{
      id: Ecto.UUID.generate(),
      season_number: sequence("season_number", &(&1 + 1)),
      start_datetime_utc: DateTime.utc_now(),
      end_datetime_utc: DateTime.utc_now() |> DateTime.add(1, :day)
    }
  end
end
