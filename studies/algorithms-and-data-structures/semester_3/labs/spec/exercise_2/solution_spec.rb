# frozen_string_literal: true

require "spec_helper"
require_relative "../../lib/exercise_2/solution"

RSpec.describe "Exercise2::Solution" do
  # Always returns 495 or 0
  # Returns 0 only for integers with same digits
  # Num of iterations is < 6
  # Num of possible combinations is limited by facts:
  # 1. results for 123, 321, 213 will be always 321 - 123
  # 1. which means that (3 * 100 + 2 * 10 + 1) - (1 * 100 + 2 * 10 + 3) etc.
  # 2. after first iteration 2 digit is always 9
  # 3. after first iteration 1 digit + 3 digit is always 9
  # 4. then there is limited number of combinations
  SAME_DIGITS = ((1..9).to_a.map { |el| "#{el}#{el}#{el}".to_i }).freeze
  SAME_DIGITS_CALC_RESULT = [0].freeze
  SAME_DIGITS_OUTPUT = 0

  DIFF_DIGITS = ((1..999).to_a - SAME_DIGITS).freeze
  DIFF_DIGITS_CALC_RESULT = [99, 198, 297, 396, 495, 594, 693, 792, 891].freeze
  DIFF_DIGITS_OUTPUT = 495

  SAME_DIGITS.each do |n|
    it "returns 0 for number - #{n}" do
      solution = Exercise2::Solution.new(n)
      result = solution.run

      solution.calculation_results.each do |el|
        expect(SAME_DIGITS_OUTPUT).to eq(el.to_i)
      end

      expect(solution.iterations).to eq(1)
      expect(result).to eq(SAME_DIGITS_OUTPUT)
    end
  end

  DIFF_DIGITS.to_a.each do |n|
    it "returns 495 for number #{n}" do
      solution = Exercise2::Solution.new(n)
      result = solution.run

      solution.calculation_results.each do |el|
        digits = el.split("").map(&:to_i)

        expect(DIFF_DIGITS_CALC_RESULT).to include(el.to_i)
        expect(digits[1]).to eq(9)
        expect(digits[0] + digits[2]).to eq(9)
      end

      expect(solution.iterations).to be <= 6
      expect(result).to eq(DIFF_DIGITS_OUTPUT)
    end
  end

  xit "with print" do
    number = rand(1..999)
    solution = Exercise2::Solution.new(number, true)
    result = solution.run

    expect(result).to eq(495)
  end
end
