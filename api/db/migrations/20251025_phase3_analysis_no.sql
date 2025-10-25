-- PHASE 3: Safer analysis_no allocation (idempotent trigger)

CREATE OR REPLACE FUNCTION public.image_analyses_allocate_no()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
  next_no integer;
BEGIN
  IF NEW.analysis_no IS NOT NULL THEN
    RETURN NEW;
  END IF;

  PERFORM pg_advisory_xact_lock(hashtextextended(NEW.workspace_id::text, 0));

  SELECT COALESCE(MAX(analysis_no),0)+1 INTO next_no
    FROM public.image_analyses
   WHERE workspace_id = NEW.workspace_id;

  NEW.analysis_no := next_no;
  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_trigger WHERE tgname='trg_image_analyses_allocate_no'
  ) THEN
    CREATE TRIGGER trg_image_analyses_allocate_no
      BEFORE INSERT ON public.image_analyses
      FOR EACH ROW EXECUTE FUNCTION public.image_analyses_allocate_no();
  END IF;
END$$;
