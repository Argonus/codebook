defmodule Chapter2Test do
  use ExUnit.Case
  doctest Chapter2

  test "greets the world" do
    assert Chapter2.hello() == :world
  end
end
