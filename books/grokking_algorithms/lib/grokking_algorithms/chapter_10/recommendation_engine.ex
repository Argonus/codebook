defmodule GrokkingAlgorithms.Chapter10.RecommendationEngine do
  @moduledoc false

  @movie_types [:comedy, :action, :drama, :horror, :romance]

  @values %{
    patricia: %{comedy: 3, action: 4, drama: 4, horror: 1, romance: 4},
    justyne: %{comedy: 4, action: 3, drama: 5, horror: 1, romance: 5},
    morpheus: %{comedy: 2, action: 5, drama: 1, horror: 3, romance: 1}
  }

  @type movies :: %{
          comendy: integer,
          action: integer,
          drama: integer,
          horror: integer,
          romance: integer
        }
  @type person :: {atom, movies}

  @spec find_nearest(person) :: person
  def find_nearest({input_name, input_points}) do
    @values
    |> Enum.map(fn {reference_name, reference_points} ->
      points =
        Enum.reduce(@movie_types, 0.0, fn type, acc ->
          a1 = Map.fetch!(input_points, type)
          a2 = Map.fetch!(reference_points, type)
          acc + :math.pow(a1 * a1 - a2 * a2, 2)
        end)

      {reference_name, :math.sqrt(points)}
    end)
    |> Enum.sort_by(
      fn {_name, points} ->
        points
      end,
      :desc
    )
  end
end
