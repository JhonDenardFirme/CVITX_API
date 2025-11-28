BEGIN;

-- 1) Add new columns (safe if already exists)
ALTER TABLE video_analyses
  ADD COLUMN IF NOT EXISTS variant TEXT NOT NULL DEFAULT 'cmt',
  ADD COLUMN IF NOT EXISTS run_id UUID,
  ADD COLUMN IF NOT EXISTS expected_snapshots INTEGER,
  ADD COLUMN IF NOT EXISTS processed_snapshots INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS processed_ok INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS processed_err INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS run_started_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS run_finished_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_snapshot_at TIMESTAMPTZ;

-- 2) snapshot_s3_key: drop NOT NULL + drop uniqueness (old schema had BOTH a constraint + extra index)
ALTER TABLE video_analyses
  ALTER COLUMN snapshot_s3_key DROP NOT NULL;

ALTER TABLE video_analyses
  DROP CONSTRAINT IF EXISTS video_analyses_snapshot_s3_key_key;

DROP INDEX IF EXISTS ux_video_analyses_snapshot;

-- 3) Backfill variant for existing rows
UPDATE video_analyses SET variant = COALESCE(variant, 'cmt');

-- 4) Deduplicate: keep the most recently updated row per (workspace_id, video_id, variant)
WITH ranked AS (
  SELECT
    id,
    ROW_NUMBER() OVER (
      PARTITION BY workspace_id, video_id, variant
      ORDER BY updated_at DESC, created_at DESC
    ) AS rn
  FROM video_analyses
)
DELETE FROM video_analyses v
USING ranked r
WHERE v.id = r.id
  AND r.rn > 1;

-- 5) Add canonical unique index
CREATE UNIQUE INDEX IF NOT EXISTS ux_video_analyses_video_variant
  ON video_analyses (workspace_id, video_id, variant);

-- 6) Create per-track detections table
CREATE TABLE IF NOT EXISTS video_detections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL REFERENCES video_analyses(id) ON DELETE CASCADE,
  run_id UUID NOT NULL,
  track_id INTEGER NOT NULL,
  snapshot_s3_key TEXT NOT NULL,
  detected_in_ms INTEGER,
  detected_at TIMESTAMPTZ,
  yolo_type TEXT,
  type_label TEXT, type_conf DOUBLE PRECISION,
  make_label TEXT, make_conf DOUBLE PRECISION,
  model_label TEXT, model_conf DOUBLE PRECISION,
  plate_text TEXT, plate_conf DOUBLE PRECISION,
  colors JSONB,
  assets JSONB,
  latency_ms INTEGER,
  memory_gb DOUBLE PRECISION,
  status TEXT NOT NULL DEFAULT 'done',
  error_msg TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (analysis_id, run_id, track_id)
);

CREATE INDEX IF NOT EXISTS ix_video_detections_run
  ON video_detections (analysis_id, run_id);

COMMIT;

