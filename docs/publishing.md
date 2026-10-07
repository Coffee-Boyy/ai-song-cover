# Publishing to GitHub

The local initial commit is on `codex/initial-pipeline`. Review the code and the explicitly included example audio before publishing. A remote repository has not been created yet.

With GitHub CLI installed and signed in, publish a reviewed copy with:

```sh
git branch -m main
gh repo create voice-cover-pipeline --public --source . --remote origin --push
```

Alternatively create an empty GitHub repository in the web interface, then:

```sh
git branch -m main
git remote add origin https://github.com/YOUR_ACCOUNT/voice-cover-pipeline.git
git push -u origin main
```

Models, full generated covers, input datasets, local profiles and the downloaded backend are ignored. The included example MP3s are intentionally tracked; their media rights are separate from the code license. No credentials are required by the cover pipeline itself.
