#!/bin/bash
cd /root/e0lab/e0
for w in d8k1 d8k2 rw8 rw1; do
  echo "=== [$(date +%H:%M)] ${w} ===" | tee -a exts_all.log
  /opt/conda/bin/python exts_baseline.py --which "${w}" --epochs 40 \
    > "exts_${w}.log" 2>&1
  echo "=== [${w}] exit=$? $(date +%H:%M) ===" | tee -a exts_all.log
done
echo "ALL-DONE $(date +%H:%M)" | tee -a exts_all.log
