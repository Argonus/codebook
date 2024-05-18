defmodule CheckersWeb.MatchControllerTest do
  use CheckersWeb.ConnCase

  import Hammox
  import Checkers.Factory

  setup :verify_on_exit!

  describe "create/2" do
    test "creates a match", %{conn: conn} do
      user = build(:user, id: 123)
      uuid = Ecto.UUID.generate()
      match = %Checkers.Matches.MatchStruct{id: uuid, host_id: 123, status: :initialized}

      expect(Checkers.MatchesMock, :create_match, fn user_id ->
        assert user_id == 123
        {:ok, match}
      end)

      authed_conn = Pow.Plug.assign_current_user(conn, user, [])
      resp_conn = post(authed_conn, "/matches", %{})

      assert resp_conn.status == 302
    end

    test "requires authentication", %{conn: conn} do
      resp_conn = post(conn, "/matches", %{})

      assert resp_conn.status == 302
    end
  end
end
