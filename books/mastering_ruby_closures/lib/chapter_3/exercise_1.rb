# frozen_string_literal: true

class String
  def to_proc
    proc { |obj, args| obj.send(self, *args) }
  end
end