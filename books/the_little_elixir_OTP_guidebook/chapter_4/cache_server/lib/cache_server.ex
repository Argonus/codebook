defmodule CacheServer do
  @moduledoc """
  Documentation for CacheServer.
  """

  def start_link(), do: CacheServer.Worker.start_link()

  def write(key, value), do: CacheServer.Worker.write(CacheServer.Worker, key, value)

  def read(key), do: CacheServer.Worker.read(CacheServer.Worker, key)

  def delete(key), do: CacheServer.Worker.delete(CacheServer.Worker, key)

  def clear, do: CacheServer.Worker.clear(CacheServer.Worker)

  def exists?(key), do: CacheServer.Worker.exists?(CacheServer.Worker, key)

  def stop, do: CacheServer.Worker.stop(CacheServer.Worker)
end
