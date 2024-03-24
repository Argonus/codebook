# frozen_string_literal: true
require_relative "./sic_list"
require_relative "./move_to_front_list"

module Exercise4
  class Solution
    BASIC_MODE = :B
    private_constant :BASIC_MODE
    MF_MODE = :MF
    private_constant :MF_MODE

    def initialize(mode)
      case mode
      when BASIC_MODE then @impl = BasicList.new
      when MF_MODE then @impl = MoveToFrontList.new
      end
    end

    def access(value)
      @impl.access(value)
    end

    def insert(value)
      @impl.insert(value)
    end

    def delete(value)
      @impl.delete(value)
    end

    def search_cost
      @impl.search_cost
    end

    def insert_cost
      @impl.insert_cost
    end

    def delete_cost
      @impl.delete_cost
    end

    def clear_costs
      @impl.clear_costs
    end

    def cost
      [search_cost, insert_cost, delete_cost].sum
    end

    def cost_details
      "#{search_cost}, #{insert_cost}, #{delete_cost}"
    end

    def to_a
      @impl.to_a
    end
  end
end
