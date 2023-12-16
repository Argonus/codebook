defmodule Chapter2.Exercise3Test do
  use ExUnit.Case, async: true

  alias Chapter2.Exercise3

  describe "transform/0" do
    test "return result" do
      assert [9, 4, 1] == Exercise3.transform()
    end
  end

  describe "pipe_transform/0" do
    test "return result" do
      assert [9, 4, 1] == Exercise3.pipe_transform()
    end
  end
end
