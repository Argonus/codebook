# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_1/exercise_4"

RSpec.describe "Exercise 3" do
  it "works with is even" do
    is_even = -> (number) { number % 2 == 0 }
    is_not_even = COMPLEMENT.call(is_even)

    expect(is_not_even.call(4)).to eq false
    expect(is_not_even.call(5)).to eq true
  end
end
