DO $$
DECLARE v_typ text;
BEGIN
  SELECT data_type INTO v_typ
  FROM information_schema.columns
  WHERE table_schema='public' AND table_name='image_analysis_results' AND column_name='colors';

  IF v_typ = 'json' THEN
    CREATE INDEX IF NOT EXISTS image_analysis_results_colors_expr_gin
      ON public.image_analysis_results
      USING gin ( (colors::jsonb) jsonb_path_ops );
  END IF;
END$$;
ANALYZE public.image_analysis_results;
