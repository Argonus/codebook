defmodule Algorithms.BinarySearch do
  @type target :: integer
  @type result :: integer

  @spec search(list, target) :: result
  def search([], _target), do: -1
  def search([target], target), do: 0

  def search(list, target) when is_list(list) and is_integer(target) do
    search(list, target, 0, length(list) - 1)
  end

  defp search([], _, _, _), do: -1
  defp search(_, _, min, max) when min > max, do: -1

  # [1, 2, 3, 4, 5]
  defp search(list, target, min, max) do
    guess = compute_guess(max, min)
    maybe_target = Enum.fetch!(list, guess)

    cond do
      # 3 = 3 => 3
      maybe_target == target -> guess
      # 1 > 3 => [1, 2]
      maybe_target > target -> search(list, target, min, guess - 1)
      # 3 < 4 => [4, 5]
      maybe_target < target -> search(list, target, guess + 1, max)
    end
  end

  defp compute_guess(max, min) do
    ((max + min) / 2) |> floor()
  end
end
