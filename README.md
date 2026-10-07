# Voice Cover Pipeline

Turn your singing reference recordings and a supplied song into an RVC voice cover. Prepare and review the references, train your own voice model, then generate covers with automatic pitch-guide repair, register continuity, pitch polish, waveform cleanup and source-adaptive studio effects.

This repository contains the pipeline and examples. Your recordings, trained model, song inputs, training logs and generated full covers stay in ignored local directories. No trained Lisa weights are distributed.

## Listen to the examples

The reference examples below are **Lisa's cleaned singing phrases from the seven supplied recordings**, concatenated chronologically with short silent separators. Talking and rejected phrases are omitted. They demonstrate the reference material used for the existing demo voice; these are listening reels, not a record-disjoint evaluation dataset.

| Reference | Audio |
| --- | --- |
| First supplied recording | [Cleaned reference audio](examples/references/original.mp3) |
| Second supplied recording | [Cleaned reference audio](examples/references/second.mp3) |
| Longer primary recording | [Cleaned reference audio](examples/references/primary.mp3) |
| Additional recording 1 | [Cleaned reference audio](examples/references/new01.mp3) |
| Additional recording 2 | [Cleaned reference audio](examples/references/new02.mp3) |
| Additional recording 3 | [Cleaned reference audio](examples/references/new03.mp3) |
| Additional recording 4 | [Cleaned reference audio](examples/references/new04.mp3) |

Generated examples:

- [Symphony — 20-second cover excerpt with adaptive studio effects](examples/covers/symphony.mp3). This existing demo uses the earlier register arrangement; it was not regenerated with the newest continuity planner.
- [Trampoline — 8-second excerpt containing the repaired pitch-guide passage](examples/covers/trampoline.mp3).

GitHub may show audio files as download links. For embedded players, clone the repository and open [the listening page](examples/listen.html) locally, or run `python -m http.server 8000` and visit `http://localhost:8000/examples/listen.html`. [Example provenance and checksums](examples/manifest.json).

## Requirements

- Python 3.11 or 3.12, Git and FFmpeg available locally.
- An [Applio](https://github.com/IAHispano/Applio) runtime and its dependencies. This pipeline pins backend commit `21392273e01da7e0169c18567d60c9b9d40282dc`; it uses scoped compatibility adapters instead of editing the upstream checkout.
- An NVIDIA GPU is recommended for training and separation. Full inference was tested on Windows with CUDA; CPU training/performance and full macOS operation are not validated here.
- Your own reference recordings and song files. Start with one singer, several clean singing takes, and coverage of the notes/registers you want to generate.

This is an RVC training workflow, not instant cloning from a single tiny reference. Training duration and reference coverage affect voice identity and naturalness. The default 200 epochs is a starting point, not a quality promise.

## Setup

Create and activate a Python environment using your preferred environment manager, then install the project and the pinned backend:

```sh
python -m pip install -e ".[audio]"
python scripts/setup_backend.py --download-assets
python -m pip install -r tools/Applio/requirements.txt
```

Install the PyTorch build appropriate to your GPU/driver using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/). The upstream requirements include their own Torch versions; resolve that environment deliberately rather than assuming a CUDA version works on every machine.

Separation can use the same environment, or a separate environment if dependency versions conflict:

```sh
python -m pip install "audio-separator==0.47.0"
```

The setup helper fetches **generic** RMVPE, ContentVec and 40 kHz HiFi-GAN pretraining resources. It records their download hashes locally. Voice-specific weights are created by training or supplied by you. Mel-band and dereverb checkpoints are downloaded by audio-separator when first used; checkpoint licenses are independent of this repository's code license.

Create your ignored local profile:

```sh
python -m voice_cover init --applio tools/Applio
python -m voice_cover doctor --training
```

`init` writes `voice.json` with paths resolved for your machine. If you use separate environments, provide `--runtime-python /path/to/audio/python` and `--separator-python /path/to/separator/python`. `--ffmpeg /path/to/ffmpeg` is available too. See [the example profile](config/example.json).

## Prepare your reference recordings

```sh
python -m voice_cover prepare --references data/take1.mp4 data/take2.wav --out data/my-voice --profile voice.json
```

This extracts vocals, reduces reverb, creates non-overlapping training candidates and writes `data/my-voice/review.html` plus `manifest.json`. Quiet and severely clipped candidates are excluded. A whole recording is reserved for validation when multiple distinct recordings are supplied; duplicates with the same content hash stay in the same group.

**Review the candidates before training.** Energy and clipping checks cannot reliably distinguish singing from talking. Disable unwanted entries in `manifest.json`, then set `approved` to `true`. To select singing ranges before slicing, use `--segments data/singing-intervals.json` with this format:

```json
{
  "take1.mp4": [[12.0, 48.0], [60.0, 90.0]],
  "take2.wav": [[0.0, 30.0]]
}
```

For already isolated, curated singing-only inputs:

```sh
python -m voice_cover prepare --references data/dry1.wav data/dry2.wav --out data/my-voice --profile voice.json --isolated --singing-only
```

`--singing-only` is your confirmation of the input content, not an automatic speech classifier. Interval selection also marks the manifest approved, so check your selections carefully. Multiple recordings provide a recording-disjoint holdout; one recording cannot provide an independent holdout in this workflow.

## Train your voice

```sh
python -m voice_cover train --manifest data/my-voice/manifest.json --name MyVoice --profile voice.json --out-profile my-voice.local.json --epochs 200 --batch-size 4 --gpu 0
```

Training uses only enabled TRAIN entries, generic pretrained RVC v2 / ContentVec / RMVPE / 40 kHz HiFi-GAN weights, and builds a TRAIN-only retrieval index. It exports a new profile pointing to your inference checkpoint and index. Working pitch limits are calibrated from TRAIN pitch labels rather than copied from Lisa's voice. Experiments require fresh names; existing checkpoints are not silently overwritten.

Supply at least three enabled TRAIN clips. The wrapper reduces the requested batch size for small datasets to meet the backend's minimum batch count. It also initializes missing backend runtime configuration and supports single-GPU training on Windows through a compatibility adapter.

Use `--plan-only` to inspect the commands without training. Check training on held-out material and listen to representative covers before choosing a checkpoint. This wrapper prevents dataset leakage into the index; it does not claim automated identity scoring or quality optimization.

## Generate a cover

```sh
python -m voice_cover generate --song data/song.wav --profile my-voice.local.json --out outputs/my-cover
```

The automatic stages are:

1. Separate the supplied song and retain its original stereo vocal for production analysis.
2. Reduce vocal reverb and obtain a source melody guide.
3. Repair unsupported guide excursions where RMVPE and independent Praat agree.
4. Choose safe registers using neighboring phrases, voiced duration and breath-gap location. Avoid brief octave excursions when a consistent register remains safe.
5. Render one continuous RVC vocal with smooth register ramps inside phrase gaps.
6. Apply guarded source-guided pitch polish and impulse/level-spike cleanup. Mild pitch-only errors do not trigger unnecessary octave retries.
7. Estimate source ambience, stereo width, dynamics and rhythm, then add generated-vocal reverb, subtle delay and stereo doubles.
8. Check duration, pitch/voicing, suspicious dropouts and WAV/decoded-MP3 true peaks; export the full mix and isolated vocal.

Effects and register decisions use song measurements, not song names or timestamps. Fractional register ramps occur in unvoiced gaps; sweeping an octave offset continuously through sung notes would change their tuning against the backing.

With already aligned stems:

```sh
python -m voice_cover generate --vocals data/dry-lead.wav --backing data/instrumental.wav --effects-reference data/original-stereo-vocal.wav --profile my-voice.local.json --out outputs/my-cover
```

Outputs include `cover.wav`, `cover.mp3`, `vocal.wav`, `vocal.mp3`, `index.html` and detailed JSON/NumPy diagnostics. Original native and cleaned vocals are retained. `--skip-effects` and `--skip-pitch-polish` are diagnostic opt-outs; `--plan-only` performs separation/analysis but skips voice generation.

A failed final audit normally stops export with `needs_review`. `--review-preview` explicitly permits a labeled preview after failed final cleanup checks. An unsafe register plan still stops for arrangement review. Diagnostic checks do not guarantee exact identity, naturalness, pronunciation or absence of every audible glitch.

## Development and verification

```sh
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q voice_cover
```

The deterministic tests need only the base package dependencies. CI runs them on Windows and Linux with Python 3.11/3.12; full GPU checks are local integration tests, not CI tests. See [validation scope](docs/validation.md) and [architecture](docs/architecture.md).

## License and publishing

Code is MIT-licensed. Example audio, song compositions and third-party model weights are separate from the code license; see [notices](NOTICE.md). Use reference recordings and songs you are entitled to process and distribute.

The repository is ready for a GitHub remote. User profiles, datasets, full generated covers, third-party checkouts and model weights are ignored. Only the explicitly included example MP3s are exempted from the media ignore rules. No automatic YouTube downloads, cloud account access or paid services are part of this pipeline.
