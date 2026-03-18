# TODO

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
- [ ] Offline replay from snapshot files (deterministic reconstruction)

## Low priority

- [ ] Multi-symbol support in ReplayEngine
- [ ] Parquet partitioning by date for faster partial reads
