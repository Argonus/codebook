defmodule Chapter2.Exercise4Test do
  use ExUnit.Case, async: true

  alias Chapter2.Exercise4

  describe "transform/0" do
    test "return result" do
      assert :crypto.hash(:md5, "Tales from the Crypt") == Exercise4.transform()
    end
  end
end
