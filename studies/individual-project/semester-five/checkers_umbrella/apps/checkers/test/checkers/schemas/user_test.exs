defmodule Checkers.Schemas.UserTest do
  use Checkers.DataCase

  alias Checkers.Repo
  alias Checkers.Schemas.User

  @valid_changeset_params %{
    login: "user",
    email: "example@example.com",
    password: "password",
    password_confirmation: "password"
  }

  describe "changeset" do
    test "with valid attributes creates user" do
      params = @valid_changeset_params

      changeset = User.changeset(%User{}, params)

      assert changeset.valid?

      user = Repo.insert!(changeset)

      assert user.login == "user"
      assert user.email == "example@example.com"
      assert user.password != "password"
    end

    for field <- [:login, :email, :password, :password_confirmation] do
      test "return error when #{field} is missing" do
        params = Map.delete(@valid_changeset_params, unquote(field))
        changeset = User.changeset(%User{}, params)

        refute changeset.valid?
        assert {"can't be blank", [validation: :required]} = changeset.errors[unquote(field)]
      end
    end
  end
end
