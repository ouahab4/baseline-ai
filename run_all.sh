#!/usr/bin/env bash
# Full pipeline: download -> clean -> train -> explain -> anomalies. About 30-40 minutes on a 4-core laptop.
set -euo pipefail
cd "$(dirname "$0")"
echo "1/6 Download data";            python -m src.download
echo "2/6 Clean and merge";          python -m src.data
echo "3/6 Train and evaluate";       python -m src.train
echo "4/6 Explain (SHAP, figures)";  python -m src.explain
echo "5/6 Anomalies and M&V";        python -m src.anomalies
echo "6/6 Website data";             python -m src.export_site
echo "Done. Start the dashboard with: streamlit run app/dashboard.py"
