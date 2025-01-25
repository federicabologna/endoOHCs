#!/bin/bash
echo "Activating endo environment"
source /share/apps/anaconda3/2021.05/bin/activate endo
echo "Beginning endo prediction processing"
python /share/luxlab/roz/endo/predict/intent_predict.py
echo "=== Processing complete ==="