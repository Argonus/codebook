# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_3/exercise_1"

RSpec.describe "Exercise 2" do
  it "calls given function" do
    array = %w[a b c d]
    array2 = array.map(&"upcase")

    expect(array2).to eq(%w[A B C D])
  end
end
