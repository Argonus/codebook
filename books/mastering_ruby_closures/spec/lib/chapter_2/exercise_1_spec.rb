# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_2/exercise_1"

RSpec.describe "Exercise 1" do
  it "works like map with integers" do
    array = [1, 2, 3, 4]
    array2 = array.map { |x| x + 1 }

    expect(array2).to eq([2, 3, 4, 5])
  end

  it "works like map with strings" do
    array = %w[look ma no for loops]
    array2 = array.map { |x| x.upcase }

    expect(array2).to eq ["LOOK", "MA", "NO", "FOR", "LOOPS"]
  end
end
