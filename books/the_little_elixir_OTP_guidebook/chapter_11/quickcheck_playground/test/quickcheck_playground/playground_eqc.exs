defmodule QuickcheckPlayground.PlaygroundEQC do
  use ExUnit.Case
  use EQC.ExUnit

  alias QuickcheckPlayground.Playground

  property "revising the lists twice yeilds the original list" do
    forall l <- list(int) do
      ensure(l |> Enum.reverse() |> Enum.reverse() == l)
    end
  end

  property "revising the lists twice yeilds the nested list" do
    forall l <- nested_list(int) do
      ensure(l |> Enum.reverse() |> Enum.reverse() == l)
    end
  end

  property "encoding is reverse of decoding" do
    forall bin <- binary do
      ensure(bin |> Base.encode64() |> Base.decode64!() == bin)
    end
  end

  property "is_sorted?" do
    forall list <- list(int) do
      ensure(list |> Enum.sort() |> Playground.is_sorted?() == true)
    end
  end

  property "splitting a string with a delimiter" do
    forall s <- string_with_commas() do
      string = to_string(s)
      :eqc.classify(
        String.contains?(string, ","),
        :string_with_commas,
        ensure String.split(string, ",") |> Enum.join(",") == string
      )
    end
  end

  def string_with_commas do
    let len <- choose(10, 20) do
      let string <- vector(len,
        frequency([{10, oneof(:lists.seq(?a, ?z))},
          {2 , ?,}])) do
        string
      end
    end
  end

  def nested_list(gen) do
    sized size do
      nested_list(size, gen)
    end
  end

  def nested_list(0, _gen), do: []
  def nested_list(size, gen) do
    lazy do
      oneof [[gen|nested_list(size-1, gen)], [nested_list(size-1, gen)]]
    end
  end
end
