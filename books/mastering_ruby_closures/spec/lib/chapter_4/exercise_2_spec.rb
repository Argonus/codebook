# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_4/lax"

RSpec.describe "Exercise 2" do
  it "works like enumerator" do
    res = 1.upto(Float::INFINITY).
      lax.
      map { |x| x*x }.
      map { |x| x+1 }.
      drop(5).
      take(2).
      to_a

    expect(res).to eq([37, 50])
  end
end
