CREATE TABLE IF NOT EXISTS cats (
  id text PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS adoptions (
  cat_id text PRIMARY KEY REFERENCES cats(id),
  name text NOT NULL CHECK (char_length(name) BETWEEN 2 AND 24),
  adopted_at timestamptz NOT NULL DEFAULT now()
);
