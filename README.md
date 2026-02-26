# C++ Voice Design (Qwen3-TTS)

## Purpose
`qwen3_tts_cpp` is a C++ library and a set of examples for speech generation using ONNX-exported Qwen3-TTS models.

Current stack:
- `QWEN3TTS::Voice` - runtime API (`load/generate/unload`)
- `QWEN3TTS::VoiceTokenizer` - tokenization of `text + instruct` into IDs
- `QWEN3TTSUTILS` - helper utilities (WAV, tensors, sampling, parsing)

## Development Note
This project was developed with `gpt-5.3-codex` and then manually reviewed/edited.

## Project Structure
- `src/voice.h`, `src/voice.cpp`  
  Core runtime API.
- `src/tokenizer.h`, `src/tokenizer.cpp`  
  Tokenizer for Qwen3-TTS prompt format.
- `src/utils.h`, `src/utils.cpp`  
  Helper functions (including `WriteWavPcm16`).
- `examples/voice_design_cli_example.cpp`  
  CLI example.
- `examples/voice_design_timing_example.cpp`  
  Multiple generations + timing stats.
- `examples/voice_design_full_profile_example.cpp`  
  Single full-profile run.
- `CMakeLists.txt`  
  Build setup for `qwen3_tts_cpp` and examples.

## Requirements
- C++17
- CMake >= 3.18
- ONNX Runtime:
  - headers (`onnxruntime_cxx_api.h`)
  - shared library (`libonnxruntime.so` / `libonnxruntime.so.*`)
- `Threads` (pthread)
- For CUDA: ONNX Runtime GPU build + working NVIDIA runtime

## ONNX Runtime Version
- Recommended / verified: `1.24.1`
- Minimum required for this model export: `>= 1.20` (IR v10 support)
- Older versions can fail at `load()` with:
  `Unsupported model IR version: 10, max supported IR version: 9`

## Expected Files in `onnx_dir`
- `prefill_builder.onnx`
- `talker_prefill_cache.onnx`
- `talker_decode_cache.onnx`
- `code_predictor_dynamic.onnx` (or step models by pattern)
- `speech_tokenizer_decode.onnx`
- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`

## Model Files
- Hugging Face repo: https://huggingface.co/abrakadobr/qwen3-tts-onnx-cpp
- Direct files page: https://huggingface.co/abrakadobr/qwen3-tts-onnx-cpp/tree/main

## Build
### Option 1: ONNX Runtime from a local installation
```bash
cmake -S . -B build \
  -DONNX_DIR=/path/to/onnxruntime
cmake --build build -j
```

### Option 2: runtime from `venv`
`CMakeLists.txt` can auto-detect runtime library from:
`venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*`

Python wheels usually do not include C++ headers, so pass include path explicitly:
```bash
cmake -S . -B build \
  -DONNX_INCLUDE_DIR=/path/to/onnxruntime/include
cmake --build build -j
```

## CMake Integration
```cmake
add_subdirectory(path/to/qwen3_tts_cpp)

target_link_libraries(my_tts_app PRIVATE qwen3_tts_cpp)
```

## Minimal API Example
```cpp
#include "voice.h"
#include "utils.h"

int main() {
  QWEN3TTS::TtsConfig cfg;
  cfg.model.path = "path/to/onnx/model";
  cfg.device = "cpu";
  cfg.intra_threads = 6;
  cfg.inter_threads = 1;

  QWEN3TTS::GenerationParams gen;
  gen.text = "This is a test sentence.";
  gen.instruct = "Speak calmly with a soft voice.";
  gen.max_steps = 160;
  gen.codec_lang = {-1} // language token id

  auto* voice = new QWEN3TTS::Voice();
  if (!voice->load(cfg)) {
    // voice->lastErrorCode() / voice->lastErrorMessage()
    delete voice;
    return 1;
  }
  auto pcm = voice->generateVoice(gen);
  if (pcm.size() == 1 && pcm[0] < 0.0f) {
    // generation error code is stored in pcm[0]
    delete voice;
    return 2;
  }
  QWEN3TTSUTILS::WriteWavPcm16("api_example.wav", pcm, 24000);
  voice->unload();
  delete voice;
  return 0;
}
```

## Language Support

The Qwen3-TTS model supports multiple languages and dialects. Each language is identified by a specific code:

| Language | Code |
|----------|------|
| `chinese` | `2055` |
| `english` | `2050` |
| `german` | `2053` |
| `italian` | `2070` |
| `portuguese` | `2071` |
| `spanish` | `2054` |
| `japanese` | `2058` |
| `korean` | `2064` |
| `french` | `2061` |
| `russian` | `2069` |
| `beijing_dialect` | `2074` |
| `sichuan_dialect` | `2062` |
| `auto` | `-1` |

These codes are used internally by the tokenizer to process text in the specified language. The "auto" option (-1) allows the model to automatically detect the language from the input text.


## Error Handling
- `Voice::load(...)` returns `bool`:
  - `true` on success
  - `false` on failure (`lastErrorCode()` / `lastErrorMessage()` provide details)
- `Voice::generateVoice(...)` returns:
  - normal PCM samples on success
  - one-element vector with negative error code on failure

### CLI Exit Codes
| Code | Meaning |
|---|---|
| `0` | success |
| `1` | usage/help error (missing required invocation) |
| `2` | invalid CLI arguments (parse/validation error) |
| `3` | runtime/model/generation error |
| `4` | output write error (WAV write failed) |

### Common Runtime Error Codes
| Code | Meaning |
|---|---|
| `-1001` | runtime is not loaded |
| `-1002` | memory info is not initialized |
| `-1101` | tokenizer/input ids error |
| `-1102` | invalid temperature |
| `-1103` | invalid top-k |
| `-1104` | invalid tail-stop settings |
| `-1105` | invalid eos_min_steps |
| `-1106` | invalid steps |
| `-1201` | no audio codes generated |
| `-1202` | all frames trimmed |
| `-1203` | predicted code out of range |
| `-1204` | failed to select first talker code |
| `-1301` | CUDA/provider related error |
| `-1302` | ONNX/decode runtime error |
| `-1303` | failed to write codes file |
| `-1401` | tokenizer load failed |
| `-1402` | tokenizer build ids failed |
| `-3001` | invalid model path in `load()` |
| `-3002` | model/session load failure |
| `-3003` | unknown load failure |
| `-3004` | requested CUDA EP is unavailable |

## CLI Example
```bash
./build/qwen3_tts_cpp_cli_example \
  --onnx-dir /path/to/model_dir \
  --text "Hello" \
  --instruct "Speak calmly." \
  --output-wav artifacts/audio/cli_example.wav \
  --max-steps 120
  --lang "english"
```

Verified run (English phrase):
```bash
./build/qwen3_tts_cpp_cli_example \
  --onnx-dir /path/to/model_dir \
  --text "welcome to qwen3 text to speech model with c++" \
  --instruct "Speak clearly, natural female voice." \
  --output-wav /tmp/qwen3tts_welcome_en.wav \
  --max-steps 80 \
  --device cpu \
  --intra-threads 2 \
  --inter-threads 1
  --lang "russian"
```

## ORT Compatibility Note
If `load()` fails with an error like:
`Unsupported model IR version: 10, max supported IR version: 9`
your ONNX Runtime version is too old for this model.

## License
This project is licensed under Apache License 2.0. See `LICENSE`.
