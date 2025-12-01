CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS video_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL,
  video_id UUID NOT NULL,
  variant TEXT NOT NULL DEFAULT 'cmt',
  run_id UUID,
  snapshot_s3_key TEXT,
  source_kind TEXT NOT NULL DEFAULT 'video',
  status TEXT NOT NULL DEFAULT 'processing',
  error_msg TEXT,
  expected_snapshots INTEGER,
  processed_snapshots INTEGER NOT NULL DEFAULT 0,
  processed_ok INTEGER NOT NULL DEFAULT 0,
  processed_err INTEGER NOT NULL DEFAULT 0,
  run_started_at TIMESTAMPTZ,
  run_finished_at TIMESTAMPTZ,
  last_snapshot_at TIMESTAMPTZ,
  latency_ms INTEGER,
  gflops DOUBLE PRECISION,
  memory_usage DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Canonical: one active container per (workspace_id, video_id, variant)
CREATE UNIQUE INDEX IF NOT EXISTS ux_video_analyses_video_variant
  ON video_analyses (workspace_id, video_id, variant);

CREATE TABLE IF NOT EXISTS video_analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL REFERENCES video_analyses(id) ON DELETE CASCADE,
  variant TEXT NOT NULL,
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
  UNIQUE (analysis_id, variant)
);

-- Per-vehicle/per-track results (one row per tracked object)
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

CREATE INDEX IF NOT EXISTS ix_video_analyses_ws_vid
  ON video_analyses (workspace_id, video_id);

-- snapshot_s3_key is no longer unique nor required; no unique index here

CREATE INDEX IF NOT EXISTS ix_video_results_analysis_variant
  ON video_analysis_results (analysis_id, variant);

CREATE INDEX IF NOT EXISTS ix_video_detections_run
  ON video_detections (analysis_id, run_id);

