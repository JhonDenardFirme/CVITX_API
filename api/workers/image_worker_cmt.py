# Auto-generated CMT wrapper to avoid 'import *' issues.
import os
from api.workers import image_worker_baseline as base

# Force CMT invariants
base.VARIANT   = "cmt"
base.QUEUE_ENV = "SQS_ANALYSIS_CMT_URL"
base.S3_SUBDIR = "cmt"
base.QUEUE_URL = os.getenv(base.QUEUE_ENV)
base.SQS_URL = base.QUEUE_URL

if __name__ == "__main__":
    base.main()
