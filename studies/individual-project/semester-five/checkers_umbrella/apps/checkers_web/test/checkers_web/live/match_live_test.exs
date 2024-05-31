defmodule CheckersWeb.MatchLiveTest do
  use CheckersWeb.ConnCase

  import Phoenix.LiveViewTest
  import Checkers.MatchesFixtures

  @create_attrs %{id: "some id"}
  @update_attrs %{id: "some updated id"}
  @invalid_attrs %{id: nil}

  defp create_match(_) do
    match = match_fixture()
    %{match: match}
  end

  describe "Index" do
    setup [:create_match]

    test "lists all matches", %{conn: conn, match: match} do
      {:ok, _index_live, html} = live(conn, ~p"/matches")

      assert html =~ "Listing Matches"
      assert html =~ match.id
    end

    test "saves new match", %{conn: conn} do
      {:ok, index_live, _html} = live(conn, ~p"/matches")

      assert index_live |> element("a", "New Match") |> render_click() =~
               "New Match"

      assert_patch(index_live, ~p"/matches/new")

      assert index_live
             |> form("#match-form", match: @invalid_attrs)
             |> render_change() =~ "can&#39;t be blank"

      assert index_live
             |> form("#match-form", match: @create_attrs)
             |> render_submit()

      assert_patch(index_live, ~p"/matches")

      html = render(index_live)
      assert html =~ "Match created successfully"
      assert html =~ "some id"
    end

    test "updates match in listing", %{conn: conn, match: match} do
      {:ok, index_live, _html} = live(conn, ~p"/matches")

      assert index_live |> element("#matches-#{match.id} a", "Edit") |> render_click() =~
               "Edit Match"

      assert_patch(index_live, ~p"/matches/#{match}/edit")

      assert index_live
             |> form("#match-form", match: @invalid_attrs)
             |> render_change() =~ "can&#39;t be blank"

      assert index_live
             |> form("#match-form", match: @update_attrs)
             |> render_submit()

      assert_patch(index_live, ~p"/matches")

      html = render(index_live)
      assert html =~ "Match updated successfully"
      assert html =~ "some updated id"
    end

    test "deletes match in listing", %{conn: conn, match: match} do
      {:ok, index_live, _html} = live(conn, ~p"/matches")

      assert index_live |> element("#matches-#{match.id} a", "Delete") |> render_click()
      refute has_element?(index_live, "#matches-#{match.id}")
    end
  end

  describe "Show" do
    setup [:create_match]

    test "displays match", %{conn: conn, match: match} do
      {:ok, _show_live, html} = live(conn, ~p"/matches/#{match}")

      assert html =~ "Show Match"
      assert html =~ match.id
    end

    test "updates match within modal", %{conn: conn, match: match} do
      {:ok, show_live, _html} = live(conn, ~p"/matches/#{match}")

      assert show_live |> element("a", "Edit") |> render_click() =~
               "Edit Match"

      assert_patch(show_live, ~p"/matches/#{match}/show/edit")

      assert show_live
             |> form("#match-form", match: @invalid_attrs)
             |> render_change() =~ "can&#39;t be blank"

      assert show_live
             |> form("#match-form", match: @update_attrs)
             |> render_submit()

      assert_patch(show_live, ~p"/matches/#{match}")

      html = render(show_live)
      assert html =~ "Match updated successfully"
      assert html =~ "some updated id"
    end
  end
end
