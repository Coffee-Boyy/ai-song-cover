# Third-party and media notices

The MIT license covers this repository's pipeline code. Upstream packages and separately downloaded weights retain their own licenses and terms:

- [Applio](https://github.com/IAHispano/Applio), pinned by revision; its source license accompanies the downloaded checkout.
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) and the independently supplied Mel-band and UVR dereverb checkpoints.
- [SwiftF0](https://github.com/lars76/swift-f0), RMVPE, ContentVec, Praat/Parselmouth, PyTorch and Pedalboard through their upstream distributions.

No backend source checkout, generic model checkpoint, trained speaker checkpoint or retrieval index is bundled in Git.

The author explicitly requested inclusion of the seven cleaned Lisa reference-audio reels and generated cover excerpts in `examples/`. These MP3s are demonstration media, separate from the MIT code license. The code license does not grant rights to the recordings, underlying song compositions or performer identity. The generated excerpts correspond to Symphony by Clean Bandit featuring Zara Larsson and Trampoline (Facading Remix); they are labeled as generated Lisa-voice examples in the README. Original song recordings and video sources are not bundled.
