# frozen_string_literal: true

require "spec_helper"
require_relative "../../lib/exercise_3/solution"

RSpec.describe "Exercise3::Solution" do
  let(:base_array) { (1..50).to_a.shuffle! }

  1000.times do |n|
    it "returns sorted array for - #{n} digits" do
      input = (1..n).to_a.shuffle!
      result = Exercise3::Solution.new(input).run

      expect(result).to eq(input.sort)
    end
  end

  it "returns sorter array with duplicates" do
    input = (base_array + base_array).shuffle!
    result = Exercise3::Solution.new(input).run

    expect(result).to eq(input.sort)
  end

  it "returns sorter array" do
    result = Exercise3::Solution.new(base_array).run

    expect(result).to eq(base_array.sort)
  end

  # Starting from 0 idx pivot require to each time compare entire array
  it "returns number of comparisons array - worst case sorted" do
    input = base_array.sort

    solution = Exercise3::Solution.new(input)
    result = solution.run

    expect(result).to eq(input.sort)
    expect(solution.comparisons).to eq(1225)
  end

  # Starting from 0 idx pivot require to each time compare entire array
  it "returns number of comparisons array - worst case sorted & reversed" do
    input = base_array.sort.reverse

    solution = Exercise3::Solution.new(input)
    result = solution.run

    expect(result).to eq(input.sort)
    expect(solution.comparisons).to eq(1225)
  end

  # Starting from pivot require to each time compare entire array
  it "returns number of comparisons array - worst case same digits" do
    input = [1] * base_array.length

    solution = Exercise3::Solution.new(input)
    result = solution.run

    expect(result).to eq(input.sort)
    expect(solution.comparisons).to eq(1225)
  end
end
