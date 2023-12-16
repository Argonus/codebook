# frozen_string_literal: true

class String
  def each_word
    arr = self.split
    arr.each { |x| yield(x) }
  end
end