# Model weights

Run `python pipeline.py fetch-weights` to download MINIMA SuperPoint-LightGlue
and CoTracker3 to the configured weight paths. SuperPoint and stock LightGlue
are stored in the workspace's Torch cache.

Downloads are verified against the URLs and SHA-256 hashes in
`configs/weights.json`. Configure existing weight files through
`minima_weights` and `cotracker_weights` in the project JSON.
