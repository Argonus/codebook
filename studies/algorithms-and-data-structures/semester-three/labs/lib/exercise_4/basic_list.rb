# frozen_string_literal: true
require_relative "./de"

module Exercise4
  class BasicList
    attr_reader :head
    attr_reader :search_cost, :insert_cost, :transpose_cost, :delete_cost

    def initialize
      @head = nil
      @search_cost = 0
      @insert_cost = 0
      @delete_cost = 0
    end

    def access(search_value)
      raise_error if self.empty?
      raise_error if search_value.nil?

      current_node = self.head
      while current_node && compare_value?(current_node, search_value)
        current_node = current_node.next
      end

      current_node.nil? ? raise_error : current_node
    end

    def insert(value)
      @insert_cost += 1
      if self.empty?
        insert_head(value)
      else
        insert_next(value)
      end
    end

    def delete(to_delete)
      raise_error if self.empty?

      current_node = self.head
      previous_node = nil

      while current_node && compare_value?(current_node, to_delete)
        previous_node = current_node
        current_node = current_node.next
      end

      current_node.nil? ? raise_error : delete_node(previous_node, current_node)
    end

    def to_a
      values = []
      return values if self.empty?
      current_node = self.head

      while current_node.next != nil
        values << current_node.value
        current_node = current_node.next
      end
      values << current_node.value

      values
    end

    def clear_costs
      @insert_cost = 0
      @search_cost = 0
      @delete_cost = 0
    end

    private

    def compare_value?(current_node, value)
      @search_cost += 1
      current_node.value != value
    end

    def next_exist?(current_node)
      @search_cost += 1
      !current_node.next.nil?
    end

    def raise_error
      raise StandardError
    end

    def insert_head(value)
      @head = Node.new(value, nil)
    end

    def insert_next(new_value)
      raise_error if self.head.value == new_value

      current_node = self.head
      new_node = Node.new(new_value, nil)

      while next_exist?(current_node)
        raise_error if self.head.value == new_node.value
        current_node = current_node.next
      end

      current_node.next = new_node
    end

    def delete_node(previous_node, current_node)
      @delete_cost += 1

      if previous_node.nil?
        @head = current_node.next
      else
        previous_node.next = current_node.next
      end
    end

    def size
      to_a.length
    end

    def empty?
      @head.nil?
    end
  end
end
