# frozen_string_literal: true

module Exercise4
  class Node
    attr_reader :value
    attr_accessor :next

    def initialize(value, next_node)
      @value = value
      @next = next_node
    end
  end
end
