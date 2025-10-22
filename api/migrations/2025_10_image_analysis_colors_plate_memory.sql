BEGIN;

-- Optional but recommended: speed color lookups like
--   WHERE colors @> '[{"base":"Red"}]'
-- Safe to run repeatedly.
CREATE INDEX IF NOT EXISTS image_analysis_results_colors_gin
  ON public.image_analysis_results
  USING gin (colors jsonb_path_ops);

-- Plate snapshot + confidence (nullable, per-variant through model_variant)
ALTER TABLE public.image_analysis_results
  ADD COLUMN IF NOT EXISTS plate_conf double precision,
  ADD COLUMN IF NOT EXISTS plate_image_s3_key text;

-- Runtime memory usage in GB (nullable)
-- (You asked to name it memory_usage.)
ALTER TABLE public.image_analysis_results
  ADD COLUMN IF NOT EXISTS memory_usage double precision;

COMMIT;
