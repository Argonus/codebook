defmodule Pow.Phoenix.RegistrationControllerTest do
  use CheckersWeb.ConnCase

  @valid_attrs %{
    login: "john",
    email: "example@example.com",
    password: "password",
    password_confirmation: "password"
  }

  describe "create/2" do
    test "creates a user and sends a confirmation email", %{conn: conn} do
      post(conn, "/registration", user: @valid_attrs)

      [user] = Checkers.Repo.all(Checkers.Schemas.User)
      assert user.login == @valid_attrs.login
      assert user.email == @valid_attrs.email
    end
  end
end
