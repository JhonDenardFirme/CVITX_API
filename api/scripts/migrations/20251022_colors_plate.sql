BEGIN;

ALTER TABLE IF EXISTS image_analysis_results
    ADD COLUMN IF NOT EXISTS plate_conf DOUBLE PRECISION NULL,
    ADD COLUMN IF NOT EXISTS plate_image_s3_key TEXT NULL,
    ADD COLUMN IF NOT EXISTS colors JSONB NULL,
    ADD COLUMN IF NOT EXISTS vehicle_image_s3_key TEXT NULL,
    ADD COLUMN IF NOT EXISTS memory_usage DOUBLE PRECISION NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'image_analysis_results_colors_gin'
    ) THEN
        EXECUTE 'CREATE INDEX image_analysis_results_colors_gin ON image_analysis_results USING GIN (colors jsonb_path_ops)';
    END IF;
END$$;

COMMIT;
