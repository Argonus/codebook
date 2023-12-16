# frozen_string_literal: true

class Array
  def each
    x = 0
    while x < self.length
      yield self[x]
      x += 1
    end
  end

  def map
    arr = []
    self.each { |x| arr << yield(x) }
    arr
  end
end