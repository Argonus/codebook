defmodule GrokkingAlgorithms.Chapter6.MangoSellersTest do
  use ExUnit.Case, async: true

  alias GrokkingAlgorithms.Chapter6.MangoSellers

  describe "mango_seller/2" do
    test "returns name if name is seller" do
      is_seller = fn _name -> true end
      {:ok, result} = MangoSellers.mango_seller(:piotr, is_seller)

      assert result == :piotr
    end

    test "returns name if name is seller and input is binary" do
      is_seller = fn _name -> true end
      {:ok, result} = MangoSellers.mango_seller("piotr", is_seller)

      assert result == :piotr
    end

    test "returns seller when seller exists" do
      is_seller = fn name ->
        name
        |> Atom.to_string()
        |> String.ends_with?("usz")
      end

      {:ok, result} = MangoSellers.mango_seller(:piotr, is_seller)
      assert result == :janusz
    end

    test "returns error when seller not found" do
      is_seller = fn name ->
        name
        |> Atom.to_string()
        |> String.ends_with?("m")
      end

      {:error, result} = MangoSellers.mango_seller(:piotr, is_seller)
      assert result == :not_found
    end
  end
end
