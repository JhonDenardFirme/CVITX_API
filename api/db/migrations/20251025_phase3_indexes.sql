-- PHASE 3: Indexes for colors & plate search (idempotent)

CREATE INDEX IF NOT EXISTS image_analysis_results_colors_gin
  ON public.image_analysis_results
  USING gin (colors jsonb_path_ops);

CREATE INDEX IF NOT EXISTS image_analysis_results_plate_tsv
  ON public.image_analysis_results
  USING gin ((to_tsvector('simple', coalesce(plate_text,''))));
