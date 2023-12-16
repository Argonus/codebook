# frozen_string_literal: true

DB = lambda do
  books = {}

  insert = -> (author, title) do
    books[author] ||= []
    books[author] << title
  end
  delete = -> (author) { books.delete(author) }
  dump = -> { books }

  {insert: insert, dump: dump, delete: delete}
end
