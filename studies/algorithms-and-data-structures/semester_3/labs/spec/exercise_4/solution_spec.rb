# frozen_string_literal: true

require "spec_helper"
require_relative "../../lib/exercise_4/solution"

RSpec.describe "Exercise4::Solution" do
  describe "basic list" do
    let!(:list) { Exercise4::Solution.new(:B) }

    describe "insert/1" do
      it "adds first element to list" do
        list.insert(1)

        expect(list.to_a).to eq([1])

        expect(list.search_cost).to eq(0)
        expect(list.insert_cost).to eq(1)
        expect(list.delete_cost).to eq(0)
      end

      it "adds next element to end of list" do
        list.insert(1)

        list.clear_costs
        list.insert(2)
        expect(list.to_a).to eq([1, 2])

        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(1)
        expect(list.delete_cost).to eq(0)
      end

      it "raises error when element already in list" do
        list.insert(1)

        expect { list.insert(1) }.to raise_error StandardError
      end
    end

    describe "access/1" do
      it "returns value from 1 size list" do
        list.insert(1)

        list.clear_costs
        result = list.access(1)

        expect(result.value).to eq(1)
        expect(result.next).to eq(nil)

        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "returns value from beginning of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        result = list.access(1)

        expect(result.value).to eq(1)
        expect(result.next.value).to eq(2)

        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "returns element from middle of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        result = list.access(2)

        expect(result.value).to eq(2)
        expect(result.next.value).to eq(3)

        expect(list.search_cost).to eq(2)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "returns element from end of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        result = list.access(3)

        expect(result.value).to eq(3)
        expect(result.next).to eq(nil)

        expect(list.search_cost).to eq(3)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "does not change order of elements" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        order_before = list.to_a
        list.access(3)

        expect(order_before).to eq list.to_a
      end

      it "raises error when list is empty" do
        expect { list.access(1) }.to raise_error StandardError
      end

      it "raises error when element is not in list" do
        list.insert(1)

        expect { list.access(2) }.to raise_error StandardError
      end
    end

    describe "delete/1" do
      it "deletes element from 1 size list" do
        list.insert(1)

        list.clear_costs
        list.delete(1)

        expect(list.to_a).to eq([])
        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "deletes element from beginning of list" do
        list.insert(1)
        list.insert(2)

        list.clear_costs
        list.delete(1)

        expect(list.to_a).to eq([2])
        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "deletes element from middle size list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        list.delete(2)

        expect(list.to_a).to eq([1, 3])
        expect(list.search_cost).to eq(2)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "deletes element from end size list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        list.delete(3)

        expect(list.to_a).to eq([1, 2])
        expect(list.search_cost).to eq(3)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "raises error when list is empty" do
        expect { list.delete(1) }.to raise_error StandardError
      end

      it "raises error when element is not in list" do
        list.insert(1)

        expect { list.delete(2) }.to raise_error StandardError
      end
    end
  end

  describe "move to front list" do
    let!(:list) { Exercise4::Solution.new(:MF) }

    describe "insert/1" do
      it "adds first element to list" do
        list.insert(1)

        expect(list.to_a).to eq([1])

        expect(list.search_cost).to eq(0)
        expect(list.insert_cost).to eq(1)
        expect(list.delete_cost).to eq(0)
      end

      it "adds new element to beginning of list" do
        list.insert(1)
        list.insert(2)

        list.clear_costs
        list.insert(3)
        expect(list.to_a).to eq([3, 2, 1])

        expect(list.search_cost).to eq(2)
        expect(list.insert_cost).to eq(3)
        expect(list.delete_cost).to eq(0)
      end

      it "raises error when element is already in list" do
        list.insert(1)

        expect { list.insert(1) }.to raise_error StandardError
      end
    end

    describe "access/1" do
      it "returns element from 1 size list" do
        list.insert(1)

        list.clear_costs
        result = list.access(1)

        expect(result.value).to eq 1
        expect(result.next).to eq nil

        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "returns element from beginning of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        result = list.access(3)

        expect(result.value).to eq(3)

        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "return element from middle of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        result = list.access(2)

        expect(result.value).to eq(2)

        expect(list.search_cost).to eq(2)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "returns element from end of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        result = list.access(1)

        expect(result.value).to eq(1)

        expect(list.search_cost).to eq(4)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(0)
      end

      it "changes order of elements in list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        expect(list.to_a).to eq([3, 2, 1])

        list.access(2)

        expect(list.to_a).to eq([2, 3, 1])
      end

      it "raises error when list is empty" do
        expect { list.access(1) }.to raise_error StandardError
      end

      it "raises error when element is not in list" do
        list.insert(1)

        expect { list.access(2) }.to raise_error StandardError
      end
    end

    describe "delete/1" do
      it "deletes value from 1 size list" do
        list.insert(1)

        list.clear_costs
        list.delete(1)

        expect(list.to_a).to eq([])

        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "deletes value from beginning of list" do
        list.insert(1)
        list.insert(2)

        list.clear_costs
        list.delete(2)

        expect(list.to_a).to eq([1])

        expect(list.search_cost).to eq(1)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "deletes value from middle of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        list.delete(2)

        expect(list.to_a).to eq([3, 1])

        expect(list.search_cost).to eq(2)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "deletes value from end of list" do
        list.insert(1)
        list.insert(2)
        list.insert(3)

        list.clear_costs
        list.delete(1)

        expect(list.to_a).to eq([3, 2])

        expect(list.search_cost).to eq(3)
        expect(list.insert_cost).to eq(0)
        expect(list.delete_cost).to eq(1)
      end

      it "raises error if element is not in list" do
        list.insert(1)

        expect { list.delete(2) }.to raise_error StandardError
      end

      it "raises error when list is empty" do
        expect { list.delete(1) }.to raise_error StandardError
      end
    end
  end

  describe "list performance" do
    let!(:list_b) { Exercise4::Solution.new(:B) }
    let!(:list_mf) { Exercise4::Solution.new(:MF) }
    let!(:file) { File.open("./test.txt", "a+") }

    let!(:input) { (0..10000).to_a.shuffle! }
    let!(:operations) { { access: 0, insert: 0, delete: 0 } }

    after(:each) do
      file.close
    end

    100.times do |n|
      it "generate report #{n}" do
        file.write("Test run #{n} \n")

        (1..1000).to_a.shuffle!.first.times do |operation_n|
          operation = operations.keys.shuffle.first
          input_idx = rand(operation_n)
          operation_input = input[input_idx]

          operations[operation] += 1
          apply_operation(list_b, operation, operation_input)
          apply_operation(list_mf, operation, operation_input)
        end

        header = "For #{operations.inspect} operations"
        puts header
        file.puts(header)

        list_b_val = "List B cost: #{list_b.cost} (#{list_b.cost_details})"
        puts list_b_val
        file.puts(list_b_val)

        list_mf_val = "List MF cost: #{list_mf.cost} (#{list_mf.cost_details})"
        puts list_mf_val
        file.puts(list_mf_val)
      end
    end
  end

  def apply_operation(list, operation, value)
    begin
      case operation
      when :access then list.access(value)
      when :insert then list.insert(value)
      when :delete then list.delete(value)
      end
    rescue
      nil
    end
  end
end
