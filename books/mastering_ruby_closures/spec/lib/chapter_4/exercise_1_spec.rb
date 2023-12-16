# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_4/lax"

RSpec.describe "Exercise 1" do
  it "works like enumerator" do
    res = 1.upto(Float::INFINITY).
      lax.
      map { |x| x*x }.
      map { |x| x+1 }.
      select { |x| x%2 == 0}.
      take(5).
      to_a

    expect(res).to eq([2, 10, 26, 50, 82])
  end
end
