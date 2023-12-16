# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_3/exercise_2"

RSpec.describe "Exercise 3" do
  it "instantiate class" do
    array = [
      ["Mel B", "Scary"],
      ["Mel C", "Sporty"],
      ["Emma B", "Baby"],
      ["Geri H", "Ginger"],
      ["Vic B", "Posh"]
    ]
    expected_results = [
      "Mel B vel Scary Spice",
      "Mel C vel Sporty Spice",
      "Emma B vel Baby Spice",
      "Geri H vel Ginger Spice",
      "Vic B vel Posh Spice"
    ]

    results = array.map(&SpiceGirl).map(&:to_s)
    expect(results).to eq(expected_results)
  end
end
