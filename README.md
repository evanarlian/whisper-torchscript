# whisper-torchscript
See original [readme](original-readme.md) and [repo](https://github.com/openai/whisper). This repo modifies Whisper a little bit to enable TorchScript.

What's new?
* TorchScript-able model.
* `kv_cache` will be passed normally instead of using hooks.
* Cannot use the existing wrapper library with this new model code.
* Some modules will be duplicated in favor of using less if elses.

Same as before:
* Original checkpoints are still valid.
* Model architecture is the same as before.

Added files:
* Scriptable model in `model2.py`
* Comparison and tests in `tests.ipynb`
* Prebuilt multilingual GPT2 tokenizer in `whisper/assets/whisper_mult_gpt2/`
