defmodule CheckersWeb.MatchControllerTest do
  use CheckersWeb.ConnCase, async: false
  import Hammox
  import Checkers.Factory

  alias Checkers.Matches.MatchStruct

  setup :verify_on_exit!

  describe "create/2" do
    test "creates a match", %{conn: conn} do
      user = build(:user, id: 123)
      uuid = Ecto.UUID.generate()
      match = %MatchStruct{id: uuid, host_id: 123, status: :initialized}

      expect(Checkers.MatchesMock, :create_match, 1, fn user_id ->
        assert user_id == 123

        {:ok, match}
      end)

      authed_conn = Pow.Plug.assign_current_user(conn, user, [])
      resp_conn = post(authed_conn, "/matches", %{})

      assert resp_conn.status == 302
    end

    test "when not authenticated, does not create match", %{conn: conn} do
      resp_conn = post(conn, "/matches", %{})

      expect(Checkers.MatchesMock, :create_match, 0, fn _user_id ->
        {:ok, nil}
      end)

      assert resp_conn.status == 302
    end
  end

  describe "join/2" do
    test "joins to existing match", %{conn: conn} do
      user = build(:user, id: 123)
      match = build(:match)

      expect(Checkers.MatchesMock, :join_match, 1, fn match_id, user_id ->
        assert match_id == match.id
        assert user_id == 123

        {:ok, MatchStruct.build_from_schema(%{match | player_id: 123})}
      end)

      authed_conn = Pow.Plug.assign_current_user(conn, user, [])
      resp_conn = patch(authed_conn, "/matches/#{match.id}/join", %{})

      assert resp_conn.status == 302
    end

    test "returns error when match not found", %{conn: conn} do
      user = build(:user, id: 123)
      match = build(:match)

      expect(Checkers.MatchesMock, :join_match, 1, fn match_id, user_id ->
        assert match_id == match.id
        assert user_id == 123

        {:ok, MatchStruct.build_from_schema(%{match | player_id: 123})}
      end)

      authed_conn = Pow.Plug.assign_current_user(conn, user, [])
      resp_conn = patch(authed_conn, "/matches/#{match.id}/join", %{})

      assert resp_conn.status == 302
    end

    test "when not authenticated, does not join to match", %{conn: conn} do
      match = build(:match)
      resp_conn = patch(conn, "/matches/#{match.id}/join", %{})

      expect(Checkers.MatchesMock, :join_match, 0, fn _match_id, _user_id ->
        {:ok, nil}
      end)

      assert resp_conn.status == 302
    end
  end
end
