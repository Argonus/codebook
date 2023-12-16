# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_1/exercise_5"

RSpec.describe "Exercise 5" do
  it "reduce" do
    args = [1, 2, 3, 4, 5]
    function = -> (x, acc) { acc << x * 2 }
    result = REDUCER.call(args, [], function)

    expect(result).to eq [2, 4, 6, 8, 10]
  end
end
