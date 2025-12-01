BEGIN;

-- 1) Extend video_analyses with V2/V3 run metadata + timing/memory/compute (idempotent)
ALTER TABLE video_analyses
  ADD COLUMN IF NOT EXISTS variant TEXT NOT NULL DEFAULT 'cmt',
  ADD COLUMN IF NOT EXISTS run_id UUID,
  ADD COLUMN IF NOT EXISTS expected_snapshots INTEGER,
  ADD COLUMN IF NOT EXISTS processed_snapshots INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS processed_ok INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS processed_err INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS run_started_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS run_finished_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_snapshot_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS latency_ms INTEGER,
  ADD COLUMN IF NOT EXISTS gflops DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS memory_usage DOUBLE PRECISION;

-- 2) snapshot_s3_key: drop NOT NULL + drop old uniqueness
ALTER TABLE video_analyses
  ALTER COLUMN snapshot_s3_key DROP NOT NULL;

ALTER TABLE video_analyses
  DROP CONSTRAINT IF EXISTS video_analyses_snapshot_s3_key_key;

DROP INDEX IF EXISTS ux_video_analyses_snapshot;

-- 3) Backfill variant for existing rows
UPDATE video_analyses
SET variant = COALESCE(variant, 'cmt');

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

-- 5) Canonical unique index for (workspace_id, video_id, variant)
CREATE UNIQUE INDEX IF NOT EXISTS ux_video_analyses_video_variant
  ON video_analyses (workspace_id, video_id, variant);

-- 6) Ensure per-track detections table exists with final schema (fresh DB case)
CREATE TABLE IF NOT EXISTS video_detections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL REFERENCES video_analyses(id) ON DELETE CASCADE,
  run_id UUID NOT NULL,
  track_id INTEGER NOT NULL,
  snapshot_s3_key TEXT NOT NULL,
  detected_in_ms INTEGER,
  detected_at TIMESTAMPTZ,
  yolo_type TEXT,
  type_label TEXT,
  type_conf DOUBLE PRECISION,
  make_label TEXT,
  make_conf DOUBLE PRECISION,
  model_label TEXT,
  model_conf DOUBLE PRECISION,
  plate_text TEXT,
  plate_conf DOUBLE PRECISION,
  parts JSONB NOT NULL DEFAULT '[]'::jsonb,
  colors JSONB,
  assets JSONB,
  latency_ms INTEGER,
  gflops DOUBLE PRECISION,
  memory_usage DOUBLE PRECISION,
  status TEXT NOT NULL DEFAULT 'done',
  error_msg TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (analysis_id, run_id, track_id)
);

CREATE INDEX IF NOT EXISTS ix_video_detections_run
  ON video_detections (analysis_id, run_id);

-- 7) Add parts to video_analysis_results (if missing)
ALTER TABLE IF EXISTS video_analysis_results
  ADD COLUMN IF NOT EXISTS parts JSONB NOT NULL DEFAULT '[]'::jsonb;

-- 8) Add parts to video_detections (if table existed before with older schema)
ALTER TABLE IF EXISTS video_detections
  ADD COLUMN IF NOT EXISTS parts JSONB NOT NULL DEFAULT '[]'::jsonb;

-- 9) Rename memory_gb -> memory_usage in video_analysis_results (or add if neither exists)
DO $$
DECLARE
  has_old BOOLEAN;
  has_new BOOLEAN;
BEGIN
  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'video_analysis_results'
      AND column_name = 'memory_gb'
  ) INTO has_old;

  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'video_analysis_results'
      AND column_name = 'memory_usage'
  ) INTO has_new;

  IF has_old AND NOT has_new THEN
    ALTER TABLE video_analysis_results RENAME COLUMN memory_gb TO memory_usage;
  ELSIF NOT has_old AND NOT has_new THEN
    ALTER TABLE video_analysis_results ADD COLUMN memory_usage DOUBLE PRECISION;
  END IF;
END $$;

-- 10) Rename memory_gb -> memory_usage in video_detections (or add if neither exists)
DO $$
DECLARE
  has_old BOOLEAN;
  has_new BOOLEAN;
BEGIN
  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'video_detections'
      AND column_name = 'memory_gb'
  ) INTO has_old;

  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'video_detections'
      AND column_name = 'memory_usage'
  ) INTO has_new;

  IF has_old AND NOT has_new THEN
    ALTER TABLE video_detections RENAME COLUMN memory_gb TO memory_usage;
  ELSIF NOT has_old AND NOT has_new THEN
    ALTER TABLE video_detections ADD COLUMN memory_usage DOUBLE PRECISION;
  END IF;
END $$;

-- 11) Ensure gflops exists on video_analysis_results
ALTER TABLE IF EXISTS video_analysis_results
  ADD COLUMN IF NOT EXISTS gflops DOUBLE PRECISION;

-- 12) Ensure gflops exists on video_detections (if table existed before with older schema)
ALTER TABLE IF EXISTS video_detections
  ADD COLUMN IF NOT EXISTS gflops DOUBLE PRECISION;

COMMIT;

