#!/bin/bash

# This script calls the python prediction script, passing along the arguments.
# 1. Trip duration in days
# 2. Miles traveled
# 3. Total amount of receipts

python3 predict.py "$1" "$2" "$3"