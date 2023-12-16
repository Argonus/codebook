# frozen_string_literal

module ActiveRecord
  class Schema
    def self.define(_version, &block)
      instance_eval(&block)
    end

    def self.create_table(table_name, options = {}, &block)
      table = Table.new(table_name, options)
      yield table
      table
    end
  end

  class Table
    attr_reader :data, :name, :options

    def initialize(name, options)
      @name = name
      @options = options
      @data = {string: [], integer: [], datetime: []}
    end

    def string(value)
      @data[:string] << value
    end

    def integer(value)
      @data[:integer] << value
    end

    def datetime(value)
      @data[:datetime] << value
    end
  end
end