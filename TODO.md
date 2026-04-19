# TODO

## Done

- [x] **Fix tick_size/step_size** — Was 0.1/0.001, actual Binance values 0.01/0.00001. Now fetched from REST API
- [x] **Investigate locked book (bid == ask)** — Caused by wrong tick_size. Fixed: spread min = 1 tick (0.01) after correction
- [x] **Default interval 50→100ms** — Matches depth@100ms stream, no synthetic grid nodes needed
- [x] **Default trade_window 100→1000ms** — 100ms too narrow, most grid nodes had zero volume
- [x] **Fix book causality** — Grid nodes emitted BEFORE apply_update, matching live system behavior
- [x] **Fix trade causality** — Only trades with recv_ts <= grid_ts passed to FeatureExtractor via deque popleft
- [x] **Fix LabelBuilder** — Keep past snapshot when future midprice is None (retry next grid node)
- [x] **Restore is_crossed to >=** — tick_size fix resolved locked book, strict check safe now
- [x] **Fetch config from Binance API** — Removed hardcoded YAML, fetch_symbol_config from /api/v3/exchangeInfo
- [x] **Dataset metadata** — Build params saved in Parquet metadata
- [x] **Overwrite protection** — CLI requires --overwrite to replace existing dataset
- [x] **Document recv_ts semantics** — All pipeline timing uses local receive time
- [x] **Pytest test harness** — 45 causal contract tests, in-memory replay, no pyarrow dependency
- [x] **Public process_event API** — Tests use public method instead of private _on_depth/_on_trade
- [x] **Snapshot-based book initialization** — Recorder fetches REST depth snapshot at startup and on gaps per Binance docs. Snapshot stored as depthSnapshot event in parquet stream. ReplayEngine bootstraps from snapshot with full U/u sequence validation. Retry if snapshot too old. Drain WS buffer after fetch to bound recv_ts error.
- [x] **Feature warmup after bootstrap** — Warmup suppresses emission (not book updates) while feature state stabilizes. warmup >= trade_window enforced. Resets on each re-bootstrap.
- [x] **3-state replay engine** — Explicit WAIT_SNAPSHOT / WARMING / LIVE state machine with named transitions. Replaced scattered boolean flags.
- [x] **BinanceStream sync tests** — 18 tests: _handle_message unit tests (recv_ts, stale filtering, overlap, gaps, trades) + _do_sync integration tests with mock WS.
- [x] **Warmup contract enforcement** — FeatureExtractor.trade_window public API, ReplayEngine raises ValueError if warmup < trade_window. Tests use same contract as production.

## High priority

- [ ] **First ML model** — Short-horizon midprice prediction baseline
- [x] **Record new data with snapshots** — Old recordings without snapshots are incompatible. Record fresh data using updated recorder.
- [x] **Compact microstructure feature pack** — OFI (3 scales), queue_delta_diff, depth_update_count, time_since_last_mid_move, signed_trade_volume, realized_vol_microprice. Added observe_depth hook in FeatureExtractor, wired through ReplayEngine.
- [x] **Economic taker evaluation notebook** — `notebooks/economically_meaningful_target_evaluation.ipynb`. 3-class bps target (2/4/6), Ridge(old)/Ridge(new)/Ridge(old+new) ablation, fee-proxy sweep (5/8/10 bps). Result on v3 (h2000/5000/10000, 1/1/1 UTC day split): best test mean signed move ≈ 1.17 bps at 5 % coverage — far below fee scale, setup not close to taker viability.

## Normal

- [ ] Collect more data (multiple days, different market regimes)
- [ ] Explore larger horizons (500ms, 1s) for label with fewer zeros
- [ ] Performance benchmarks (events/sec throughput)
- [ ] Cross-stream ordering safety margin for trades
- [ ] Synchronize time horizon for raw data and dataset in research notebook

## Low priority

- [ ] Multi-symbol support in ReplayEngine
- [ ] Parquet partitioning by date for faster partial reads
