#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
target = ROOT / "video_analysis" / "worker_utils" / "common.py"

text = target.read_text(encoding="utf-8")

# Replace the whole start_video_run block up to (but not beyond) get_video_run
pattern = r"def start_video_run\([^)]*\)[\s\S]*?def get_video_run"

replacement = '''def start_video_run(workspace_id: str, video_id: str, variant: str, run_id: str) -> str:
    """
    Ensure there is exactly ONE video_analyses row per (workspace_id, video_id, variant).

    - If none exists: insert a new row in 'running' state and return its analysis_id.
    - If one already exists: reset + reuse it for this run (update counters, status, run_id, timestamps)
      and return the existing analysis_id.

    This respects the UNIQUE index:
        ux_video_analyses_video_variant (workspace_id, video_id, variant)
    and avoids psycopg2.errors.UniqueViolation on re-runs.
    """
    if not run_id:
        run_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc)

    with _video_conn() as conn:
        # Check if a row already exists for this (workspace, video, variant)
        existing = conn.execute(
            text(
                """
                SELECT id
                FROM video_analyses
                WHERE workspace_id = :wid
                  AND video_id     = :vid
                  AND variant      = :variant
                LIMIT 1
                """
            ),
            {"wid": workspace_id, "vid": video_id, "variant": variant},
        ).mappings().first()

        if existing is None:
            analysis_id = str(uuid.uuid4())
            conn.execute(
                text(
                    """
                    INSERT INTO video_analyses (
                        id,
                        workspace_id,
                        video_id,
                        variant,
                        run_id,
                        status,
                        expected_snapshots,
                        processed_snapshots,
                        processed_ok,
                        processed_err,
                        run_started_at,
                        run_finished_at,
                        last_snapshot_at,
                        error_msg,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :id,
                        :wid,
                        :vid,
                        :variant,
                        :run_id,
                        'running',
                        0,
                        0,
                        0,
                        0,
                        :now,
                        NULL,
                        NULL,
                        NULL,
                        :now,
                        :now
                    )
                    """
                ),
                {
                    "id": analysis_id,
                    "wid": workspace_id,
                    "vid": video_id,
                    "variant": variant,
                    "run_id": run_id,
                    "now": now,
                },
            )
            return analysis_id

        analysis_id = str(existing["id"])

        conn.execute(
            text(
                """
                UPDATE video_analyses
                SET
                    run_id             = :run_id,
                    status             = 'running',
                    expected_snapshots = 0,
                    processed_snapshots = 0,
                    processed_ok       = 0,
                    processed_err      = 0,
                    run_started_at     = :now,
                    run_finished_at    = NULL,
                    last_snapshot_at   = NULL,
                    error_msg          = NULL,
                    latency_ms         = NULL,
                    memory_usage       = NULL,
                    updated_at         = :now
                WHERE id = :id
                """
            ),
            {"id": analysis_id, "run_id": run_id, "now": now},
        )

        return analysis_id


def get_video_run(workspace_id: str, video_id: str, variant: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest run for (workspace, video, variant) or None.

    Used by main_worker to decide if it should attach to an existing run
    or bootstrap a new run via start_video_run().
    """
    with _video_conn() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                  id,
                  workspace_id,
                  video_id,
                  variant,
                  run_id,
                  status,
                  expected_snapshots,
                  processed_snapshots,
                  processed_ok,
                  processed_err,
                  run_started_at,
                  run_finished_at,
                  last_snapshot_at,
                  error_msg
                FROM video_analyses
                WHERE workspace_id = :wid
                  AND video_id     = :vid
                  AND variant      = :variant
                  AND run_id IS NOT NULL
                ORDER BY updated_at DESC,
                         run_started_at DESC NULLS LAST,
                         created_at DESC
                LIMIT 1
                """
            ),
            {"wid": workspace_id, "vid": video_id, "variant": variant},
        ).mappings().first()

    if not row:
        return None

    return {
        "analysis_id": str(row["id"]),
        "workspace_id": str(row["workspace_id"]),
        "video_id": str(row["video_id"]),
        "variant": row["variant"],
        "run_id": str(row["run_id"]) if row["run_id"] else None,
        "status": row["status"],
        "expected_snapshots": row["expected_snapshots"],
        "processed_snapshots": row["processed_snapshots"],
        "processed_ok": row["processed_ok"],
        "processed_err": row["processed_err"],
        "run_started_at": row["run_started_at"],
        "run_finished_at": row["run_finished_at"],
        "last_snapshot_at": row["last_snapshot_at"],
        "error_msg": row["error_msg"],
    }
'''

new_text, count = re.subn(pattern, replacement, text, count=1)

if count != 1:
    print(f"[patch] ERROR: expected to patch 1 block, but patched {count}. Aborting.", file=sys.stderr)
    sys.exit(1)

target.write_text(new_text, encoding="utf-8")
print("[patch] Successfully updated start_video_run in", target)
