-- Parent table: one per uploaded image
CREATE TABLE IF NOT EXISTS public.image_analyses (
  id uuid PRIMARY KEY,
  workspace_id uuid NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
  analysis_no integer NOT NULL, -- 1,2,3… per workspace
  title text,
  description text,
  input_image_s3_key text NOT NULL,  -- original upload key
  content_type text NOT NULL,        -- image/jpeg, image/png, ...
  size_bytes bigint NOT NULL,
  status text CHECK (status IN ('uploaded','queued','processing','done','error')) NOT NULL DEFAULT 'uploaded',
  error_msg text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(workspace_id, analysis_no),
  UNIQUE(workspace_id, input_image_s3_key)
);

-- Enum for result variant
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_variant') THEN
    CREATE TYPE model_variant AS ENUM ('baseline','cmt');
  END IF;
END$$;

-- Child table: two rows per analysis (baseline & cmt)
CREATE TABLE IF NOT EXISTS public.image_analysis_results (
  id uuid PRIMARY KEY,
  analysis_id uuid NOT NULL REFERENCES public.image_analyses(id) ON DELETE CASCADE,
  workspace_id uuid NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
  model_variant model_variant NOT NULL,   -- 'baseline' | 'cmt'

  type text,           type_conf double precision,
  make text,           make_conf double precision,
  model text,          model_conf double precision,

  parts jsonb,                                   -- [{name, conf}, ...]
  colors jsonb,                                  -- ["red","black"] optional
  plate_text text,                               -- optional

  annotated_image_s3_key text,                   -- optional; annotated image
  thresholds jsonb,                              -- e.g. {"make_model_min":0.7}
  evidence jsonb,                                -- raw bboxes/logits/etc
  latency_ms integer,
  gflops double precision,
  status text CHECK (status IN ('ready','error')) NOT NULL DEFAULT 'ready',
  error_msg text,
  created_at timestamptz NOT NULL DEFAULT now(),

  UNIQUE(analysis_id, model_variant)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS image_analysis_results_ws
  ON public.image_analysis_results(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS image_analysis_results_variant
  ON public.image_analysis_results(model_variant);

-- Optional text-search index for plate_text (simple)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE c.relname='image_analysis_results_plate_tsv' AND n.nspname='public'
  ) THEN
    CREATE INDEX image_analysis_results_plate_tsv
      ON public.image_analysis_results
      USING gin (to_tsvector('simple', coalesce(plate_text,'')));
  END IF;
END$$;
