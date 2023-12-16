# frozen_string_literal: true

require "spec_helper"
require_relative "../../../lib/chapter_1/exercise_3"

RSpec.describe "Exercise 3" do
  it "works with insert" do
    db = DB.call
    eagles = ["Eagles", "Hell Freezes Over"]
    pink_floyd = ["Pink Floyd", "The Wall"]

    expect(db[:insert].call(*eagles)).to eq ["Hell Freezes Over"]
    expect(db[:insert].call(*pink_floyd)).to eq ["The Wall"]
  end

  it "works with dump" do
    db = DB.call
    eagles1 = ["Eagles", "Hell Freezes Over"]
    eagles2 = ["Eagles", "Life in the Fast Lane"]
    pink_floyd = ["Pink Floyd", "The Wall"]

    db[:insert].call(*eagles1)
    db[:insert].call(*eagles2)
    db[:insert].call(*pink_floyd)

    expect(db[:dump].call).to eq({"Eagles" => ["Hell Freezes Over", "Life in the Fast Lane"], "Pink Floyd" => ["The Wall"]})
  end

  it "works with delete" do
    db = DB.call
    eagles1 = ["Eagles", "Hell Freezes Over"]
    eagles2 = ["Eagles", "Life in the Fast Lane"]
    pink_floyd = ["Pink Floyd", "The Wall"]

    db[:insert].call(*eagles1)
    db[:insert].call(*eagles2)
    db[:insert].call(*pink_floyd)

    expect(db[:delete].call("Pink Floyd")).to eq(["The Wall"])
    expect(db[:dump].call).to eq({"Eagles" => ["Hell Freezes Over", "Life in the Fast Lane"]})
  end
end
