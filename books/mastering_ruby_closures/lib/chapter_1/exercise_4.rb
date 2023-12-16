# frozen_string_literal: true

COMPLEMENT = lambda do |predicate|
  lambda do |args|
    !predicate.call(args)
  end
end
