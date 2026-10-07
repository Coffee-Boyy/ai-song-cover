# Validation scope

## Verified locally

- 46 passing deterministic register, harmonic-guide, impulse/level cleanup, dataset split, interval, backend-adapter and profile-path tests.
- A complete six-second supplied-song run through separation, foreground extraction, inference, pitch polish, cleanup, source-adaptive effects and WAV/MP3 export, using the freshly cloned pinned upstream backend and existing local speaker weights.
- Reference preparation on synthetic harmonic recordings, including a recording-disjoint validation group.
- A complete one-epoch GPU training smoke test on synthetic harmonic recordings: preprocessing, feature extraction, four optimizer steps, inference checkpoint export, TRAIN-only retrieval-index creation and calibrated profile export. Validation recordings stayed outside training and the index.

GPU/audio environments were existing working environments on Windows; these checks do not establish that installing every dependency from scratch works on every platform. CI tests do not train a model or generate an audio cover.

## Quality limits

Pitch/voicing and true-peak checks can detect specific failure modes but do not certify exact voice identity, pronunciation, perceptual quality or word alignment. Isolation may damage singing or retain music. Energy filters cannot distinguish talking from singing, so training manifests require explicit approval or explicit singing-only selection.

The included Lisa examples were generated using an existing trained model with substantial prior tuning. The public 200-epoch training preset has not been shown to reproduce that result for arbitrary speakers. The successful synthetic one-epoch smoke test validates the training/export path only and says nothing about voice likeness.
