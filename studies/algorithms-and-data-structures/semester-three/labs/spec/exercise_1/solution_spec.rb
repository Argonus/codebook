# frozen_string_literal: true

require "spec_helper"
require_relative "../../lib/exercise_1/solution"

RSpec.describe "Exercise1" do
  describe "for static input" do
    let!(:input) { rand(1..100) }

    10.times do |n|
      it "returns always ball black - #{n}" do
        solution = Exercise1::Solution.new(input * n).run
        expect(solution).to eq(:black)
      end

      it "invariant check and succeeded checks is the same - #{n}" do
        solution = Exercise1::Solution.new(input * n)
        solution.run
        expect(solution.invariant_checks).to eq(solution.invariant_checks_succeeded)
      end
    end
  end
end
