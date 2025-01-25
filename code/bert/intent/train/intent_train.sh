#!/bin/bash
echo "Activating endo environment"
source /share/apps/anaconda3/2021.05/bin/activate endo
echo "Beginning endo training processing"
python /share/luxlab/roz/endo/train/intent_train.py
echo "=== Processing complete ==="