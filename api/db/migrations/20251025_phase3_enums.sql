-- PHASE 3: CHECK constraints for statuses (idempotent)

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='image_analyses_status_chk') THEN
    ALTER TABLE public.image_analyses
      ADD CONSTRAINT image_analyses_status_chk
      CHECK (status IN ('uploaded','queued','processing','done','error'));
  END IF;
END$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='image_analysis_results_status_chk') THEN
    ALTER TABLE public.image_analysis_results
      ADD CONSTRAINT image_analysis_results_status_chk
      CHECK (status IN ('ready','error'));
  END IF;
END$$;
