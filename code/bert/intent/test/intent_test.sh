#!/bin/bash
echo "Activating endo environment"
source /share/apps/anaconda3/2021.05/bin/activate endo
echo "Beginning endo testing processing"
python /share/luxlab/roz/endo/test/intent_test.py
echo "=== Processing complete ==="