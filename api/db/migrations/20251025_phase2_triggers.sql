-- PHASE 2: Status transitions & idempotency helpers (idempotent)

CREATE OR REPLACE FUNCTION public.image_analysis_results_set_status_default()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW.status IS NULL THEN
    IF NEW.error_msg IS NOT NULL AND NEW.error_msg <> '' THEN
      NEW.status := 'error';
    ELSE
      NEW.status := 'ready';
    END IF;
  END IF;
  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_results_status_default') THEN
    CREATE TRIGGER trg_results_status_default
      BEFORE INSERT ON public.image_analysis_results
      FOR EACH ROW EXECUTE FUNCTION public.image_analysis_results_set_status_default();
  END IF;
END$$;

CREATE OR REPLACE FUNCTION public.image_analysis_parent_status_bump()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
  v_ready_count int;
  v_any_error boolean;
  v_parent_status text;
BEGIN
  IF NEW.status = 'error' THEN
    UPDATE public.image_analyses
       SET status    = 'error',
           error_msg = CASE
                         WHEN error_msg IS NULL OR error_msg = '' THEN COALESCE(NEW.error_msg,'worker error')
                         ELSE error_msg || E'\n' || COALESCE(NEW.error_msg,'worker error')
                       END
     WHERE id = NEW.analysis_id AND status <> 'error';
    RETURN NULL;
  END IF;

  SELECT status INTO v_parent_status FROM public.image_analyses WHERE id = NEW.analysis_id;
  IF v_parent_status IN ('uploaded','queued') THEN
    UPDATE public.image_analyses SET status = 'processing'
     WHERE id = NEW.analysis_id AND status IN ('uploaded','queued');
  END IF;

  SELECT COUNT(*) FILTER (WHERE status = 'ready'),
         BOOL_OR(status = 'error')
    INTO v_ready_count, v_any_error
    FROM public.image_analysis_results
   WHERE analysis_id = NEW.analysis_id;

  IF (NOT COALESCE(v_any_error,false)) AND v_ready_count >= 2 THEN
    UPDATE public.image_analyses SET status = 'done'
     WHERE id = NEW.analysis_id AND status <> 'error';
  END IF;

  RETURN NULL;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_parent_status_bump') THEN
    CREATE TRIGGER trg_parent_status_bump
      AFTER INSERT OR UPDATE OF status, error_msg
      ON public.image_analysis_results
      FOR EACH ROW EXECUTE FUNCTION public.image_analysis_parent_status_bump();
  END IF;
END$$;
