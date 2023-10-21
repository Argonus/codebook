# frozen_string_literal: true
# Implement Merge Sort

module Exercise3
  class Solution
    attr_reader :comparisons

    def initialize(input)
      @input = input
      @comparisons = 0
    end

    def run
      do_sort(@input)
    end

    private

    def do_sort(input)
      return input if input.length <= 1

      pivot = input[0]
      smaller, greater = partition_by(input[1..-1], pivot)
      do_sort(smaller) + [pivot] + do_sort(greater)
    end

    def partition_by(array, pivot)
      smaller, greater = [], []

      array.each do |n|
        n < pivot ? smaller << n : greater << n
        @comparisons += 1
      end

      [smaller, greater]
    end
  end
end
