CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS video_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL,
  video_id UUID NOT NULL,
  snapshot_s3_key TEXT UNIQUE NOT NULL,
  source_kind TEXT NOT NULL DEFAULT 'snapshot',
  status TEXT NOT NULL DEFAULT 'processing',
  error_msg TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS video_analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id UUID NOT NULL REFERENCES video_analyses(id) ON DELETE CASCADE,
  variant TEXT NOT NULL,
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
  UNIQUE (analysis_id, variant)
);

CREATE INDEX IF NOT EXISTS ix_video_analyses_ws_vid
  ON video_analyses (workspace_id, video_id);

CREATE UNIQUE INDEX IF NOT EXISTS ux_video_analyses_snapshot
  ON video_analyses (snapshot_s3_key);

CREATE INDEX IF NOT EXISTS ix_video_results_analysis_variant
  ON video_analysis_results (analysis_id, variant);
