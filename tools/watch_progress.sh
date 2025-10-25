#!/usr/bin/env bash
set -euo pipefail
: "${AWS_REGION:?}"; : "${BUCKET:?}"; : "${WORKSPACE_ID:?}"; : "${VIDEO_ID:?}"
FRAME_STRIDE="${FRAME_STRIDE:-6}"

TMPVID="/tmp/watch_src_$VIDEO_ID.mp4"
RAW_KEY="demo_user/$WORKSPACE_ID/$VIDEO_ID/raw/$(aws s3 ls s3://$BUCKET/demo_user/$WORKSPACE_ID/$VIDEO_ID/raw/ --region "$AWS_REGION" | awk '{print $4}' | head -n1)"

# Get frame count quickly (download small; OpenCV reads locally)
if [ ! -s "$TMPVID" ]; then
  aws s3 cp "s3://$BUCKET/$RAW_KEY" "$TMPVID" --region "$AWS_REGION" >/dev/null
fi

python - <<'PY' > /tmp/_total_frames.txt
import cv2, sys
p="/tmp/watch_src_'"$VIDEO_ID"'.mp4".replace("''","")
cap=cv2.VideoCapture(p)
n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(n)
PY

TOTAL_FRAMES=$(cat /tmp/_total_frames.txt)
if [ "$TOTAL_FRAMES" -gt 0 ]; then
  TOTAL_ITERS=$(( (TOTAL_FRAMES + FRAME_STRIDE - 1) / FRAME_STRIDE ))
else
  TOTAL_ITERS=0
fi

echo "Total frames: $TOTAL_FRAMES  stride: $FRAME_STRIDE  est iters: $TOTAL_ITERS"

START=$(date +%s)
LAST_CNT=0
while true; do
  CNT=$(aws s3 ls "s3://$BUCKET/demo_user/$WORKSPACE_ID/$VIDEO_ID/snapshots/" --region "$AWS_REGION" | wc -l | tr -d ' ')
  NOW=$(date +%s)
  ELAP=$((NOW-START))
  DELTA=$((CNT-LAST_CNT))
  RATE=$(python - <<PY
import sys
cnt=int(sys.argv[1]); elap=int(sys.argv[2])
print("{:.2f}".format(cnt/elap if elap>0 else 0))
PY
  $CNT $ELAP)

  if [ "$TOTAL_ITERS" -gt 0 ]; then
    REM=$(( TOTAL_ITERS - CNT )); REM=$(( REM<0 ? 0 : REM ))
    ETA=$(python - <<PY
import sys
rem=int(sys.argv[1]); rate=float(sys.argv[2])
sec = int(rem/rate) if rate>0 else -1
print(sec)
PY
    $REM $RATE)
    # nice hh:mm:ss
    H=$(printf "%02d" $(( (ETA/3600) )))
    M=$(printf "%02d" $(( (ETA%3600)/60 )))
    S=$(printf "%02d" $(( ETA%60 )))
    ETA_TXT=$([ "$ETA" -gt 0 ] && echo "${H}:${M}:${S}" || echo "?")
    printf "[watch] iters=%s/%s  rate=%s/s  elapsed=%02d:%02d:%02d  eta=%s\n" "$CNT" "$TOTAL_ITERS" "$RATE" $((ELAP/3600)) $(((ELAP%3600)/60)) $((ELAP%60)) "$ETA_TXT"
  else
    printf "[watch] iters=%s  rate=%s/s  elapsed=%02d:%02d:%02d\n" "$CNT" "$RATE" $((ELAP/3600)) $(((ELAP%3600)/60)) $((ELAP%60))
  fi
  LAST_CNT=$CNT
  sleep 10
done
