# frozen_string_literal: true

require "prawn"
require "fileutils"
require 'prawn/emoji'

class CardGenerator
  def initialize(cards_dir: "./cards", output_dir: "./pdf_cards")
    @cards_dir = cards_dir
    @output_dir = output_dir
    FileUtils.mkdir_p(@output_dir)
  end

  def generate_all_pdfs
    card_files = Dir.glob(File.join(@cards_dir, "*.txt"))
                    .sort_by do |p|
                      name = File.basename(p, ".txt")
                      [name.end_with?("-EN") ? 1 : 0, name.sub(/-EN\z/, '')]
                    end

    if card_files.empty?
      puts "No card files found in #{@cards_dir}. Please run WeddingBingo.new.generate first."
      return
    end

    out_path = File.join(@output_dir, "all_cards.pdf")
    puts "Found #{card_files.length} cards. Building booklet => #{out_path}"

    Prawn::Document.generate(out_path, page_size: 'A4', margin: 20) do |pdf|
      font_path = File.expand_path("DejaVuSans.ttf", File.dirname(__FILE__))
      use_external_font = false
      begin
        pdf.font font_path
        use_external_font = true
      rescue
        pdf.font "Helvetica"
      end

      # Per-page layout (two rows: one card above the other)
      gutter = 20
      full_w = pdf.bounds.width
      half_h = (pdf.bounds.height - gutter) / 2.0

      card_files.each_slice(2).with_index do |pair, idx|
        pair.each_with_index do |card_file, row|
          next unless card_file
          card_name = File.basename(card_file, ".txt")
          lang = card_name.end_with?("-EN") ? "EN" : "PL"
          display_card_name = card_name.sub(/-EN\z/, '').tr("_", "/")
          grid = read_grid(card_file)

          # Top or bottom box
          x = pdf.bounds.left
          y = row.zero? ? pdf.bounds.top : pdf.bounds.top - half_h - gutter

          pdf.bounding_box([x, y], width: full_w, height: half_h) do
            render_card(pdf, grid, display_card_name, use_external_font, lang)
          end
        end

        # Dotted cut line between the two cards on the page
        pdf.dash(1, space: 3)
        cut_y = pdf.bounds.top - half_h - (gutter / 2.0) - 8
        pdf.stroke_horizontal_line pdf.bounds.left, pdf.bounds.right, at: cut_y
        pdf.undash

        # New page except after last
        pdf.start_new_page unless idx == (card_files.each_slice(2).count - 1)
      end
    end

    puts "Booklet generated at: #{out_path}"
  end

  # Helper to read grid from a txt card file
  def read_grid(card_file)
    File.readlines(card_file, chomp: true).map { |line| line.split(" | ") }
  end

  # Render a single card inside current bounding box
  def render_card(pdf, grid, display_card_name, use_external_font, lang)
    # Language badge in top-right corner
    badge_w = 24
    badge_h = 12
    pdf.bounding_box([pdf.bounds.right - badge_w, pdf.bounds.top], width: badge_w, height: badge_h) do
      pdf.fill_color 'eeeeee'
      pdf.fill_rectangle [0, badge_h], badge_w, badge_h
      pdf.fill_color '000000'
      pdf.font_size 8
      pdf.text lang, align: :center, valign: :center
    end
    # Localized labels
    if lang == "EN"
      label_card   = "Card"
      label_names  = "Names:"
      label_coupons= "Coupons:"
      label_footer = "Fill the card and win prizes!"
    else
      label_card   = "Karta"
      label_names  = "Imiona:"
      label_coupons= "Kupony:"
      label_footer = "Wypełnij kartę i wygraj nagrody!"
    end

    # Title
    pdf.font_size 20
    pdf.text "Piotr i Karolina", align: :center
    pdf.move_down 6

    # Card number
    pdf.font_size 11
    pdf.text "#{label_card}: #{display_card_name}", align: :center
    pdf.move_down 10

    # Names
    pdf.font_size 10
    pdf.text label_names, align: :left
    pdf.move_down 4
    pdf.stroke_horizontal_line 0, pdf.bounds.width, at: pdf.cursor - 3
    pdf.move_down 14

    # Grid sizing
    remaining_height = pdf.cursor - 60
    cell_w = pdf.bounds.width / 5.0
    cell_h = [remaining_height / 5.0, 45].min

    grid_start_y = pdf.cursor
    5.times do |r|
      5.times do |c|
        x = c * cell_w
        y = grid_start_y - (r * cell_h)
        pdf.stroke_rectangle [x, y], cell_w, cell_h

        cell_text = grid[r][c]

        if r == 2 && c == 2
          # Center tile (middle message)
          pdf.font_size 8
          pdf.fill_color "eeeeee"
          pdf.fill_rectangle [x + 1, y - 1], cell_w - 2, cell_h - 2
          pdf.fill_color "000000"
          pdf.bounding_box([x + 3, y - 3], width: cell_w - 6, height: cell_h - 6) do
            pdf.text cell_text, align: :center, valign: :center, overflow: :shrink_to_fit
          end
        else
          # Question text smaller and top-aligned
          pdf.font_size 6
          question_area = (cell_h * 0.55)
          pdf.bounding_box([x + 3, y - 3], width: cell_w - 6, height: question_area - 2) do
            pdf.text cell_text, align: :center, valign: :top, overflow: :shrink_to_fit
          end

          # Single dotted answer line near the bottom
          pdf.dash(1, space: 2)
          answer_y = y - cell_h + 6
          pdf.stroke_horizontal_line x + 6, x + cell_w - 6, at: answer_y
          pdf.undash
        end
      end
    end

    pdf.move_cursor_to(grid_start_y - (cell_h * 5))
    pdf.move_down 10

    # Footer
    pdf.font_size 8
    pdf.text label_footer, align: :center

    # Coupons (one row of 10)
    pdf.move_down 10
    pdf.font_size 10
    pdf.text label_coupons, align: :left
    pdf.move_down 6

    coupon_w = pdf.bounds.width / 10.0
    coupon_h = 32
    row_y = pdf.cursor
    10.times do |i|
      x = i * coupon_w
      # frame
      pdf.stroke_rectangle [x, row_y], coupon_w, coupon_h

      # emoji / reward (raised a bit, centered)
      reward = case (i + 1)
               when 1,2,3,5,9 then "🍷/ ☕️"
               when 4,6,7 then "☕️ & ⚙️"
               when 8,10 then "🥇"
               else "" end
      pdf.text_box reward,
                   at: [x, row_y - 4], width: coupon_w, height: coupon_h - 14,
                   align: :center, valign: :center, size: 10

      # card id (bottom area, raised slightly)
      pdf.text_box display_card_name,
                   at: [x, row_y - coupon_h + 10], width: coupon_w, height: 8,
                   align: :center, valign: :bottom, size: 6
    end
  end

  private

  def sanitize_text(text)
    # Replace Polish characters with ASCII equivalents for PDF compatibility
    text.gsub(/ą/, 'a')
        .gsub(/ć/, 'c')
        .gsub(/ę/, 'e')
        .gsub(/ł/, 'l')
        .gsub(/ń/, 'n')
        .gsub(/ó/, 'o')
        .gsub(/ś/, 's')
        .gsub(/ź/, 'z')
        .gsub(/ż/, 'z')
        .gsub(/Ą/, 'A')
        .gsub(/Ć/, 'C')
        .gsub(/Ę/, 'E')
        .gsub(/Ł/, 'L')
        .gsub(/Ń/, 'N')
        .gsub(/Ó/, 'O')
        .gsub(/Ś/, 'S')
        .gsub(/Ź/, 'Z')
        .gsub(/Ż/, 'Z')
  end

  def generate_pdf_for_card(card_file)
    card_name = File.basename(card_file, ".txt")
    display_card_name = card_name.tr("_", "/")
    pdf_path = File.join(@output_dir, "#{card_name}.pdf")
    
    # Read the card content
    grid = []
    File.readlines(card_file, chomp: true).each do |line|
      grid << line.split(" | ")
    end
    
    # Generate PDF
    Prawn::Document.generate(pdf_path) do |pdf|
      font_path = File.expand_path("DejaVuSans.ttf", File.dirname(__FILE__))
      begin
        pdf.font font_path
        use_external_font = true
        puts "Successfully loaded DejaVuSans.ttf font"
      rescue => e
        puts "Could not load DejaVuSans.ttf: #{e.message}"
        puts "Using Helvetica with sanitized text"
        pdf.font "Helvetica"
        use_external_font = false
      end
      
      # Wedding title
      pdf.font_size 24
      title_text = use_external_font ? "Piotr i Karolina" : sanitize_text("Piotr i Karolina")
      if use_external_font
        pdf.text title_text, align: :center  # No style with external fonts
      else
        pdf.text title_text, align: :center, style: :bold
      end
      pdf.move_down 10
      
      # Card number
      pdf.font_size 14
      card_text = use_external_font ? "Karta: #{display_card_name}" : sanitize_text("Karta: #{display_card_name}")
      pdf.text card_text, align: :center
      pdf.move_down 15
      
      # Names section
      pdf.font_size 12
      names_text = use_external_font ? "Imiona:" : sanitize_text("Imiona:")
      if use_external_font
        pdf.text names_text, align: :left  # No style with external fonts
      else
        pdf.text names_text, align: :left, style: :bold
      end
      pdf.move_down 5
      
      # Draw one line for names
      pdf.stroke_horizontal_line 0, pdf.bounds.width, at: pdf.cursor - 5
      pdf.move_down 20
      
      pdf.move_down 10
      
      # Calculate cell dimensions for remaining space
      remaining_height = pdf.cursor - 80 # Leave more space for footer
      cell_width = pdf.bounds.width / 5
      cell_height = [remaining_height / 5, 60].min # Ensure reasonable cell height
      
      # Draw the bingo grid
      grid_start_y = pdf.cursor
      
      5.times do |row|
        5.times do |col|
          x = col * cell_width
          y = grid_start_y - (row * cell_height)
          
          # Draw cell border
          pdf.stroke_rectangle [x, y], cell_width, cell_height
          
          # Add text to cell
          cell_text = use_external_font ? grid[row][col] : sanitize_text(grid[row][col])
          
          # Special formatting for middle cell
          if row == 2 && col == 2
            pdf.font_size 9
            pdf.fill_color "eeeeee"
            pdf.fill_rectangle [x + 1, y - 1], cell_width - 2, cell_height - 2
            pdf.fill_color "000000"
          else
            pdf.font_size 7
          end
          
          # Text positioning and wrapping
          pdf.bounding_box([x + 3, y - 3], width: cell_width - 6, height: cell_height - 6) do
            pdf.text cell_text, align: :center, valign: :center, overflow: :shrink_to_fit
          end
        end
      end
      
      # Move cursor down after the grid
      pdf.move_cursor_to(grid_start_y - (cell_height * 5))
      pdf.move_down 20
      
      # Add footer
      pdf.font_size 10
      footer_text = "Wypełnij kartę i wygraj nagrody!"
      if use_external_font
        pdf.text footer_text, align: :center  # No style with external fonts
      else
        pdf.text footer_text, align: :center, style: :italic
      end
      
      # Add coupons section
      pdf.move_down 20
      pdf.font_size 12
      coupons_text = use_external_font ? "Kupony:" : sanitize_text("Kupony:")
      if use_external_font
        pdf.text coupons_text, align: :left  # No style with external fonts
      else
        pdf.text coupons_text, align: :left, style: :bold
      end
      pdf.move_down 10
      
      # Draw 10 coupons in one row
      coupon_width = pdf.bounds.width / 10
      coupon_height = 40
      coupons_start_y = pdf.cursor
      
      10.times do |col|
        coupon_number = col + 1
        x = col * coupon_width
        y = coupons_start_y
        
        # Draw coupon border
        pdf.stroke_rectangle [x, y], coupon_width, coupon_height
        
        # Determine reward emoji based on coupon number
        reward_emoji = case coupon_number
                      when 1, 2, 3, 5, 9
                        "🍷/ ☕️"
                      when 4, 6, 7
                        "☕️& 🔨"
                      when 8, 10
                        "🥇"
                      else
                        "Prize"
                      end
        
        # Add coupon text using absolute positioning
        pdf.font_size 8
        
        # Coupon number at top
        pdf.draw_text "#{coupon_number}", at: [x + coupon_width/2 - 5, y - 8]
        
        # Emoji/reward text in middle
        emoji_text = use_external_font ? reward_emoji : case coupon_number
                                                       when 1, 2, 3, 5, 9
                                                         "W/C"  # Wine/Coffee abbreviated
                                                       when 4, 6, 7
                                                         "C&T"  # Coffee & Tools abbreviated
                                                       when 8, 10
                                                         "GOLD" # Gold Prize abbreviated
                                                       else
                                                         "Prize"
                                                       end
        pdf.font_size 6
        pdf.draw_text emoji_text, at: [x + coupon_width/2 - 8, y - 20]
        
        # Card name at bottom
        pdf.font_size 5
        pdf.draw_text "#{display_card_name}", at: [x + 2, y - 35]
      end
      
      # Move cursor down after coupons
      pdf.move_down coupon_height + 10
    end
    
    puts "Generated: #{pdf_path}"
  end
end

# Run the generator if this file is executed directly
if __FILE__ == $0
  generator = CardGenerator.new
  generator.generate_all_pdfs
end
