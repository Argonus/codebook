# frozen_string_literal: true

class MyFile < File
  def self.open(name, mode, &block)
    begin
      file = new(name, mode)
      return file unless block_given?
      yield(file)
    ensure
      file.close
    end
  end
end