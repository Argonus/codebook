# frozen_string_literal: true

require_relative "../../../mastering_ruby_closures/chapter_2/exercise_4"

RSpec.describe "my active record schema" do
  it "works like map with strings" do

    table = ActiveRecord::Schema.define(1) do ||
      create_table("microposts", force: true) do |t|
        t.integer :id
        t.string :first_name
        t.string :last_name
        t.datetime :created_at
      end
    end

    expect(table.name).to eq("microposts")
    expect(table.options).to eq({force: true})
    expect(table.data).to eq({string: [:first_name, :last_name], integer: [:id], datetime: [:created_at]})
  end
end
