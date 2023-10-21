module Exercise2
  class Solution
    attr_reader :numbers, :iterations, :calculation_results

    def initialize(number, with_print = false)
      @n = number
      @calculation_results = []
      @iterations = 0
      @with_print = with_print
    end

    def run
      while @n != 495 && @n != 0

        # main loop calculations
        n0 = @n
        min, max = calculate_min_max(@n)
        @n1 = max
        @n2 = min
        @n = @n1 - @n2
        @iterations += 1

        # used for debug
        print_iteration(max, min, n0) if @with_print
        store_iteration_result
      end

      @n
    end

    private

    def number_plus_10(n)
      output = []

      while n < 1000
        output << n
        n = n + 10
      end

      output
    end

    def store_iteration_result
      @calculation_results << prepare_output_digits(@n)
    end

    def print_iteration(max, min, n0)
      puts "#{@iterations}- Number: #{prepare_output_digits(n0)}: #{prepare_output_digits(max)} - #{prepare_output_digits(min)} = #{prepare_output_digits(@n)}"
    end

    def calculate_min_max(number)
      calculate_permutation(number).minmax
    end

    def calculate_permutation(number)
      digits = calculate_digits(number)
      digits.permutation.to_a.map { |e| e.join("").to_i }
    end

    def calculate_digits(number)
      digits = number.digits.reverse

      case digits.length
      when 3 then digits
      when 2 then digits.unshift(0)
      when 1 then digits.unshift(0).unshift(0)
      else raise StandardError, "Invalid Integer"
      end
    end

    def prepare_output_digits(number)
      calculate_digits(number).join("")
    end
  end
end
