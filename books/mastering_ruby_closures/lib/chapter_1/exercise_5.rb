# frozen_string_literal: true

REDUCER = lambda do |args, acc, function|
  reducer = lambda do |args, acc|
    if args == []
      acc
    else
      head = args.first
      tail = args.drop(1)
      acc = function.call(head, acc)

      reducer.call(tail, acc)
    end
  end

  reducer.call(args, acc)
end
