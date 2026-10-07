# Architecture

The lightweight `voice-cover` dispatcher separates configuration, reference preparation, training and inference. Profiles resolve relative paths against their own location; local paths and private weights are not part of published examples.

`prepare` creates candidates and a review manifest. Distinct recording hashes define the TRAIN/validation split. `train` stages enabled TRAIN files only, invokes the pinned upstream preprocess/extract/train/index programs with argument arrays, checks their outputs, and writes a new voice profile. Silence fillers are disabled during extraction so they do not enter the retrieval index.

`generate` performs separation, foreground preparation, source pitch guard, location-aware register planning, continuous decoding, guarded pitch polish, waveform cleanup, source-adaptive effects and export checks. `register`, `pitch_guard` and `cleanup` provide the deterministic algorithms covered in CI.

Tracked backend source remains unmodified. Setup creates its ignored runtime configuration when missing, preserving existing settings. The training bootstrap avoids distributed-network initialization for single-GPU training. Scoped inference adapters select FLOAT native WAV output and bounded-query exact Torch retrieval, avoiding an observed Windows native Faiss-search crash. Native model architecture and weights are unchanged. For large reference banks exact retrieval can be expensive; configure `retrieval_backend: "faiss"` to use upstream retrieval on a compatible installation. Faiss is still required to read/write the retrieval index.

Studio processing estimates a source tail decay and calibrates the available reverb's impulse response to a nearby decay setting. Send levels follow measured late energy. Stereo layering uses generated-voice copies with short delays; a quiet tempo-linked echo is an approximation, not proof of the original delay settings. Source singer waveforms are not mixed into these effect sends. Independent harmony lines are not recreated.

No song-title conditions, song timestamps, Lisa training data, or Lisa model paths control the public pipeline. The included listening examples document earlier project outputs; they are not run-time configuration or universal quality benchmarks.
