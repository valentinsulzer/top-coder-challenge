# Plan of Attack: Reverse-Engineering the Legacy Reimbursement System

This document outlines the plan to reverse-engineer ACME Corp's legacy travel reimbursement system based on the provided `README.md`, `PRD.md`, `INTERVIEWS.md`, and `public_cases.json` data.

The core methodology is to start with a simple, verifiable baseline and incrementally add complexity based on data analysis and clues from the employee interviews. The `./eval.sh` script will be used as a constant feedback mechanism to measure progress.

## Phase 1: Basic Scaffolding & Evaluation Harness

The immediate goal is to create a working end-to-end pipeline that can be tested against the evaluation script. This ensures the basic mechanics are in place before tackling the complex logic.

1.  **Create `run.sh`:** Copy `run.sh.template` to `run.sh` and make it executable. This script will be the entry point for the evaluation harness.
2.  **Modify `run.sh`:** The script will be updated to execute the Python program, passing the command-line arguments `trip_duration_days`, `miles_traveled`, and `total_receipts_amount` to the Python script.
3.  **Update `src/main.py`:**
    - Refactor the `main` function to accept the three required command-line arguments.
    - Implement a dead-simple, placeholder calculation logic. A good starting point, based on the interviews, is a basic per diem and mileage rate: `reimbursement = (days * 100) + (miles * 0.58)`.
    - Ensure the final output is a single number, rounded to two decimal places, printed to standard output.
4.  **Baseline Evaluation:** Run `./eval.sh` to confirm the setup is working correctly and to establish a baseline score. This first score will likely be poor, but it verifies that the input/output format is correct.

## Phase 2: Data Exploration & Feature Engineering

With the scaffolding in place, the next step is to analyze the `public_cases.json` data to find patterns and validate the theories from the interviews.

1.  **Data Loading:** Create a script or notebook (e.g., `analysis.ipynb`) to load `public_cases.json` into a pandas DataFrame.
2.  **Feature Engineering:** Based on the interviews, create new features that are likely to be important for the calculation:
    - `miles_per_day` (the "efficiency" metric mentioned by Kevin and Marcus).
    - `receipts_per_day` (mentioned by Kevin).
3.  **Analysis & Visualization:**
    - Analyze the relationships between the inputs, engineered features, and the reimbursement amount.
    - Create plots to visualize these relationships to confirm or deny theories (e.g., plot `reimbursement` vs. `trip_duration_days` to look for the 5-day "sweet spot").
    - Look for tiers, thresholds, and non-linear patterns.

## Phase 3: Incremental Logic Implementation & Refinement

This is the core iterative loop of the project. Implement logic changes one by one, using `./eval.sh` to measure the impact of each change.

1.  **Per Diem Logic:**
    - Implement the base $100/day rate (Lisa).
    - Add the bonus for 5-day trips (Lisa).
    - Test for a broader 4-6 day "sweet spot" (Jennifer).
2.  **Mileage Logic:**
    - Implement the tiered mileage system: a higher rate for the first ~100 miles, then a lower rate (Lisa).
    - Model the "efficiency" bonus/penalty curve based on `miles_per_day`. Test Kevin's theory of an 180-220 miles/day sweet spot.
3.  **Receipt Logic (Most Complex):**
    - Implement the penalty for very low receipt amounts (Lisa, Dave).
    - Model the diminishing returns for high receipt totals.
    - Incorporate Kevin's `receipts_per_day` thresholds, which vary by trip length.
    - Add the "rounding bug" for receipt amounts ending in `.49` or `.99` (Lisa).
4.  **Iterate and Test:** After each piece of logic is added, run `./eval.sh` and check the score. If the score improves, keep the change. If not, revise or discard it.

## Phase 4: Advanced Logic & Interaction Effects

Once the primary factors are modeled, address the more complex and subtle theories.

1.  **Calculation Paths:** Implement Kevin's theory of multiple calculation paths. Use conditional logic (`if/elif/else`) to segment trips (e.g., short & high-mileage vs. long & low-mileage) and apply different formulas to each segment.
2.  **Interaction Effects:** Introduce terms into the formulas that combine factors, such as `trip_duration_days * miles_per_day`, as suggested by Kevin.
3.  **Red Herrings:** Theories about timing (day of the week, lunar cycles) are likely red herrings, as the system does not receive the date as an input. These will be ignored unless strong evidence emerges from the data itself.

## Phase 5: Finalization & Submission

1.  **Code Cleanup:** Refactor `src/main.py` for clarity and add comments explaining the final logic, referencing the interviews that inspired it.
2.  **Final Test:** Run `./eval.sh` one last time to confirm the final score on the public data.
3.  **Generate Private Results:** Run `./generate_results.sh` to produce the `private_results.txt` file for submission.
