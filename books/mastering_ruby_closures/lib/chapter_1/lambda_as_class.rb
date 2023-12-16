# frozen_string_literal: true

Counter = lambda do
  x = 0
  get_x = -> { x }
  incr = -> { x += 1 }
  decr = -> { x -= 1 }

  { get_x: get_x, incr: incr, decr: decr }
end

c1 = Counter.call

puts c1[:get_x].call
# => 0
c1[:incr].call
puts c1[:get_x].call
# => 1

Counter2 = lambda do
  x = 0
  get_x = -> { x }
  incr = -> { x += 1 }
  decr = -> { x -= 1 }

  { x: x, get_x: get_x, incr: incr, decr: decr }
end

c2 = Counter2.call

puts c2[:get_x].call
puts c2[:x]
# => 0
c2[:incr].call
puts c2[:get_x].call
# => 1
puts c2[:x]
# => 0

