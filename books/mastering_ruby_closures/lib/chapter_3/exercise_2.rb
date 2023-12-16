# frozen_string_literal: true

class SpiceGirl
  def initialize(name, nick)
    @name = name
    @nick = nick
  end

  def to_s
    "#{@name} vel #{@nick} Spice"
  end

  def self.to_proc
    proc { |obj| new(obj[0], obj[1]) }
  end
end