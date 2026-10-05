# Decisions and ambiguities

1. The supplied corrected project proposal takes precedence over generic course examples, as requested. The two DOCX files are copied into `docs/` for provenance.
2. PPG-DaLiA distributions differ in packaging and label keys. The loader therefore discovers supported subject files and key aliases instead of hardcoding a Kaggle slug. Unsupported layouts fail with an informative message.
3. The proposal specifies reference HR alignment but not one universal file schema. `dalia.py` exposes an explicit `SubjectRecord` interface and records missing/ambiguous alignment rather than silently shifting labels.
4. Peak prominence and optional preprocessing thresholds are parameters selected on validation data by the caller and persisted in the resolved configuration. Defaults are conservative and are not test-tuned.
5. The proposal gives parameter counts and formulas but no runnable checkpoints or empirical results. Repository outputs therefore use `not_run`/`not_available` until data and training are supplied.
6. The decoder CLI reconstructs a decoder-only model from the saved decoder state dict plus the configured latent dimension; the encoder and proxy are not needed for free generation.
