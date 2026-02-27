#!/bin/zsh
WORKSPACE=/Users/erikgustafsson/.openclaw/workspace/projects
HUB=/Users/erikgustafsson/.openclaw/workspace/projects/hub-cloud
mkdir -p $HUB/data/{ml_signals,canslim,breadth,risk_factors,crypto,smart_money}
cp $WORKSPACE/intermarket-ratio-ml/data/signals/signal_log_meta.csv $HUB/data/ml_signals/ 2>/dev/null && echo "✓ ml_signals" || echo "✗ ml_signals (not found)"
cp $WORKSPACE/canslim-sepa/output/composite_rankings.csv $HUB/data/canslim/ 2>/dev/null && echo "✓ canslim" || echo "✗ canslim (not found)"
cp $WORKSPACE/sp500-breadth/data/breadth_index_weekly_with_wow.csv $HUB/data/breadth/ 2>/dev/null && echo "✓ breadth_index" || echo "✗ breadth_index (not found)"
cp $WORKSPACE/sp500-breadth/data/breadth_sector_weekly.csv $HUB/data/breadth/ 2>/dev/null && echo "✓ breadth_sector" || echo "✗ breadth_sector (not found)"
cp $WORKSPACE/market-risk-factors/data/latest_scores.json $HUB/data/risk_factors/ 2>/dev/null && echo "✓ risk_factors json" || echo "✗ risk_factors json (not found)"
cp $WORKSPACE/market-risk-factors/data/composite_timeseries.parquet $HUB/data/risk_factors/ 2>/dev/null && echo "✓ risk_factors parquet" || echo "✗ risk_factors parquet (not found)"
cp $WORKSPACE/crypto-analyzer/data/history.parquet $HUB/data/crypto/ 2>/dev/null && echo "✓ crypto" || echo "✗ crypto (not found)"
echo "Data synced at $(date)"
