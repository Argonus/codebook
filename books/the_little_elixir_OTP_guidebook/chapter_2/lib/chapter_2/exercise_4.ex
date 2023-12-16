defmodule Chapter2.Exercise4 do
  @moduledoc false
  @string "Tales from the Crypt"

  @spec transform() :: string
  def transform do
    "#{(:crypto.hash(:md5, @string))}"
  end
end
