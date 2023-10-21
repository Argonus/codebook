module Exercise1
  class Solution
    attr_reader :input_size, :invariant_checks, :invariant_checks_succeeded

    def initialize(input_size)
      @invariant_checks = 0
      @invariant_checks_succeeded = 0
      @balls = prepare_balls(input_size)
      @input_size = @balls.size
    end

    def run
      raise StandardError unless balls_num_is_odd?

      while @balls.length > 1
        invariant_check

        # Drops two samples from array
        sample_1 = fetch_sample
        sample_2 = fetch_sample

        # Return black if balls have different color
        @balls.push(:black) if sample_1 != sample_2

        invariant_check
      end

      @balls.first
    end

    private

    def prepare_balls(n)
      whites = [:white] * n
      blacks = n.odd? ? [:black] * n : [:black] * (n + 1)
      whites + blacks
    end

    def fetch_sample
      idx = rand(@balls.size)
      @balls.delete_at(idx)
    end

    def invariant_check
      @invariant_checks += 1
      @invariant_checks_succeeded += 1 if balls_num_is_odd?
    end

    def balls_num_is_odd?
      @balls.filter { |el| el.eql?(:black) }.size.odd?
    end
  end
end
