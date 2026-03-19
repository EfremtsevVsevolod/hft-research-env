# TODO

## Done

- [x] **Fix tick_size/step_size** — Was 0.1/0.001, actual Binance values 0.01/0.00001. Now fetched from REST API
- [x] **Investigate locked book (bid == ask)** — Caused by wrong tick_size. Fixed: spread min = 1 tick (0.01) after correction
- [x] **Default interval 50→100ms** — Matches depth@100ms stream, no synthetic grid nodes needed
- [x] **Default trade_window 100→1000ms** — 100ms too narrow, most grid nodes had zero volume

## High priority

- [ ] **Snapshot-based book initialization** — Replace warmup with proper Binance order book management per [docs](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams#how-to-manage-a-local-order-book-correctly):
  - Record depth snapshots (`GET /api/v3/depth?limit=5000`) alongside the diff stream
  - On startup: load snapshot, set `last_update_id = lastUpdateId`
  - Discard buffered events where `u <= lastUpdateId`
  - First applied event must have `lastUpdateId` within `[U, u]`
  - On sequence gap: fetch a new snapshot instead of warmup restart
  - Remove warmup logic from `ReplayEngine`

## Normal

- [ ] Property-based tests for OrderBook, FeatureExtractor, ReplayEngine
- [ ] Performance benchmarks (events/sec throughput)
- [ ] Collect more data (multiple days, different market regimes)
- [ ] Explore larger horizons (500ms, 1s) for label with fewer zeros

## Low priority

- [ ] Multi-symbol support in ReplayEngine
- [ ] Parquet partitioning by date for faster partial reads
