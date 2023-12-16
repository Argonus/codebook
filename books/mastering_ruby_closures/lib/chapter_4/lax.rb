# frozen_string_literal: true

class Lax < Enumerator
  def initialize(receiver)
    super() do |yielder|
      begin
        receiver.each do |val|
          if block_given?
            yield(yielder, val)
          else
            yielder << val
          end
        end
      rescue StopIteration
      end
    end
  end

  def map(&block)
    Lax.new(self) do |yielder, val|
      yielder << block.call(val)
    end
  end

  def select(&block)
    Lax.new(self) do |yielder, val|
      yielder << val if block.call(val)
    end
  end

  def take(n)
    taken = 0
    Lax.new(self) do |yielder, val|
      if taken < n
        yielder << val
        taken += 1
      else
        raise StopIteration
      end
    end
  end

  def drop(n)
    dropped = 0
    Lax.new(self) do |yielder, val|
      if dropped < n
        dropped += 1
      else
        yielder << val
      end
    end
  end
end

module Enumerable
  def lax
    Lax.new(self)
  end
end

# e = Enumerator.new do |yielder|
#   [1, 2, 3].each do |val|
#     yielder << val
#     puts "Val: #{val}"
#     puts "Self: #{self}"
#     puts "Yielder: #{yielder.to_s}"
#   end
# end
#
# puts "E: #{e}"
# puts "E.next = #{e.next}"
# puts "E.next = #{e.next}"
#
# f = Fiber.new do
#   x = 0
#   loop do
#     puts "X: #{x}"
#     Fiber.yield x
#     x += 1
#   end
# end
#
# puts "F: #{f}"
# f.resume