defmodule CacheServerTest do
  use ExUnit.Case
  doctest CacheServer

  test "greets the world" do
    assert CacheServer.hello() == :world
  end
end
