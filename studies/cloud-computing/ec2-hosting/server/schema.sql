CREATE TABLE IF NOT EXISTS cats (
  id text PRIMARY KEY
);

ALTER TABLE cats ADD COLUMN IF NOT EXISTS batch_id text NOT NULL DEFAULT 'initial';
ALTER TABLE cats ADD COLUMN IF NOT EXISTS name text;
ALTER TABLE cats ADD COLUMN IF NOT EXISTS appearance text;
ALTER TABLE cats ADD COLUMN IF NOT EXISTS position smallint;

CREATE INDEX IF NOT EXISTS cats_batch_id_idx ON cats (batch_id);

CREATE TABLE IF NOT EXISTS shelter (
  id boolean PRIMARY KEY CHECK (id = true),
  batch_id text NOT NULL
);

CREATE TABLE IF NOT EXISTS adoptions (
  cat_id text PRIMARY KEY REFERENCES cats(id),
  name text NOT NULL CHECK (char_length(name) BETWEEN 2 AND 24),
  adopted_at timestamptz NOT NULL DEFAULT now()
);
