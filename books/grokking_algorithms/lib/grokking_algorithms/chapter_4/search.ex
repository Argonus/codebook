defmodule GrokkingAlgorithms.Chapter4.Search do
  @moduledoc false

  @spec search(list, integer) :: integer | nil
  def search([], _target), do: nil
  def search([target], target), do: 0
  def search([_], _target), do: nil

  def search(list, target) when is_list(list) and is_integer(target) do
    do_search(list, target, 0, length(list) - 1)
  end

  defp do_search([], _target, _min, _max), do: nil
  defp do_search(_list, _target, min, max) when min > max, do: nil

  defp do_search(list, target, min, max) do
    guess = compute_guess(min, max)
    maybe_target = Enum.fetch!(list, guess)

    cond do
      maybe_target == target -> guess
      maybe_target > target -> do_search(list, target, min, guess - 1)
      maybe_target < target -> do_search(list, target, guess + 1, max)
    end
  end

  defp compute_guess(max, min), do: ((max + min) / 2) |> floor()
end
