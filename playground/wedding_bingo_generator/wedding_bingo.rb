# frozen_string_literal: true

require "set"
require "fileutils"

class WeddingBingo
  COMPOSITION = { "w" => 2, "z" => 12, "a" => 8, "rnd" => 2 }.freeze

  def initialize(cards = 60, rows = 5, columns = 5,
                 questions_path: "./questions/pl.txt",
                 questions_en_path: "./questions/en.txt",
                 out_dir: "./cards",
                 middle: "Dzięki, że jesteście z nami! ❤️")
    @cards, @rows, @columns = Integer(cards), Integer(rows), Integer(columns)
    @middle         = middle
    @questions_path = questions_path
    @questions_en_path = questions_en_path
    @out_dir        = out_dir
    @card_numbers   = Set.new
    @questions      = load_questions
    @must_have = "Wpisz się do księgi gości."
    FileUtils.mkdir_p(@out_dir)
  end

  def generate
    # Default/current language batch
    @cards.times do
      card_number = generate_card_number
      grid        = build_grid
      save_card(card_number, grid)
    end

    # Additional 5 English cards (if questions_en.txt exists)
    orig_q_path    = @questions_path
    orig_questions = @questions
    orig_middle    = @middle
    begin
      en_path = @questions_en_path || File.join(File.dirname(@questions_path), "questions_en.txt")
      if File.exist?(en_path)
        @questions_path = en_path
        @questions      = load_questions
        @middle         = "Thank you for being with us! ❤️"

        5.times do
          card_number = generate_card_number
          grid        = build_grid
          save_card("#{card_number}-EN", grid)
        end
      else
        warn "[WeddingBingo] EN file not found: #{en_path}. Skipping English cards."
      end
    ensure
      @questions_path = orig_q_path
      @questions      = orig_questions
      @middle         = orig_middle
    end
  end

  private

  def save_card(card_number, grid)
    path = File.join(@out_dir, sanitize_filename("#{card_number}.txt"))
    File.open(path, "w") do |f|
      grid.each { |row| f.puts row.join(" | ") }
    end
  end

  def build_grid
    w = sample_unique(@questions["w"], COMPOSITION["w"])
    z = sample_unique(@questions["z"], COMPOSITION["z"])
    a = sample_unique(@questions["a"], COMPOSITION["a"])

    used = Set.new(w + z + a)
    pool = (@questions["w"] + @questions["z"] + @questions["a"]).reject { |q| used.include?(q) }.uniq
    rnd  = sample_unique(pool, COMPOSITION["rnd"])

    cells = (w + z + a + rnd).shuffle

    grid = Array.new(@rows) { Array.new(@columns) }
    mid_r, mid_c = @rows / 2, @columns / 2
    grid[mid_r][mid_c] = @middle

    it = cells.each
    @rows.times do |r|
      @columns.times do |c|
        next if r == mid_r && c == mid_c
        grid[r][c] = it.next
      end
    end
    # Ensure @must_have is present somewhere on the card (any non-center cell)
    if @must_have && !@must_have.empty?
      flat = grid.flatten
      unless flat.include?(@must_have)
        spots = []
        @rows.times do |r|
          @columns.times do |c|
            next if r == mid_r && c == mid_c
            spots << [r, c]
          end
        end
        r, c = spots.sample
        grid[r][c] = @must_have
      end
    end
    grid
  end

  def sample_unique(pool, n)
    pool.sample(n)
  end

  def load_questions
    raw = File.readlines(@questions_path, chomp: true)
              .map(&:strip)
              .reject(&:empty?)
    cats = { "w" => [], "z" => [], "a" => [] }
    raw.uniq.each do |line|
      if (m = line.match(/^\s*\[([wza])\]\s*(.+)$/i))
        cats[m[1].downcase] << m[2].strip
      end
    end
    cats
  end

  def generate_card_number
    loop do
      code = "WB/#{rand_letter}/#{rand(10..99)}"
      unless @card_numbers.include?(code)
        @card_numbers << code
        return code
      end
    end
  end

  def rand_letter
    ("A".."Z").to_a.sample
  end

  def sanitize_filename(name)
    name.gsub(/[^\w\.\-]/, "_")
  end
end

WeddingBingo.new.generate
