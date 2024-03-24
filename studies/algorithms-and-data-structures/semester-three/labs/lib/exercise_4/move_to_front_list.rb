# frozen_string_literal: true
require_relative "./sic_list"

module Exercise4
  class MoveToFrontList < BasicList
    def access(search_value)
      raise_error if self.empty?
      current_node = self.head
      previous_node = nil

      while current_node && compare_value?(current_node, search_value)
        previous_node = current_node
        current_node = current_node.next
      end

      if current_node.nil?
        raise_error
      else
        cost = calc_transpose_cost(current_node)
        result = transpose_and_return(previous_node, current_node)
        @search_cost += cost if cost >= 0

        result
      end
    end

    private

    def insert_next(new_value)
      raise_error if self.head.value == new_value
      current_node = self.head
      new_node = Node.new(new_value, head)

      while next_exist?(current_node)
        raise_error if self.head.value == new_node.value
        current_node = current_node.next
      end

      @insert_cost += size
      @head = new_node
    end

    def transpose_and_return(previous_node, current_node)
      return current_node if previous_node.nil?

      previous_node.next = current_node.next
      current_node.next = self.head
      @head = current_node
    end

    def calc_transpose_cost(current_node)
      self.to_a.find_index(current_node.value) - 1
    end
  end
end
