from workers.image_worker_baseline import *  # re-use everything
VARIANT = "cmt"
SQS_URL = os.getenv("SQS_ANALYSIS_CMT_URL")
