# crit-engines

The source of the inference engines built with [crit](https://github.com/crit-labs/crit) and
listed in [crit-hub](https://github.com/crit-labs/crit-hub). Each one was exported from a sealed
crit candidate.

The engines built before crit's plugin architecture (adapter engines, which crit no longer installs or
serves) were removed; they are rebuilt as plugin packages and return here as they are exported.

crit-hub is where to read about them: every engine's README there states its model, design,
measured results, checks and limits, and how to build it and open chat, and its `card.yaml` pins
the commit of this repository that holds its source. Start from crit-hub's
[engines/](https://github.com/crit-labs/crit-hub/tree/main/engines), then build from the commit the
engine pins.

## Layout

```text
<name>/              the engine's name in crit-hub, engines/<owner>/<name>/
  engine/            source, configuration and required runtime data
  THIRD_PARTY.md     dependency credits
  LICENSES.txt       applicable license texts and notices
```

Model weights, builds, binaries and logs stay outside the repository; the [.gitignore](.gitignore)
uses a whitelist for published files.

## License

Original content is licensed under [Apache-2.0](LICENSE). Third-party components retain their own
terms; see each engine's `THIRD_PARTY.md` and `LICENSES.txt`.
