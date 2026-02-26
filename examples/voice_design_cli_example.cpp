#include "voice.h"
#include "utils.h"

#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
namespace {

void PrintUsage(const char* exe) {
  std::cerr
      << "Usage:\n  " << exe
      << " --onnx-dir <onnx_dir> --text <text> --instruct <instruct>"
      << " [--output-wav PATH] [--save-codes-file PATH] [--max-steps N]"
      << " [--prefill-builder-file NAME] [--talker-prefill-file NAME] [--talker-decode-file NAME]"
      << " [--speech-tokenizer-file NAME] [--cp-dynamic-file NAME] [--cp-step-pattern PATTERN]"
      << " [--tokenizer-vocab-file NAME] [--tokenizer-merges-file NAME] [--tokenizer-config-file NAME]"
      << " [--ort-opt disable|basic|extended|all] [--intra-threads N] [--inter-threads N]"
      << " [--device cpu|cuda] [--prefill-device auto|cpu|cuda] [--talker-device auto|cpu|cuda]"
      << " [--cp-device auto|cpu|cuda] [--vocoder-device auto|cpu|cuda]"
      << " [--gpu-device-id N] [--gpu-mem-limit-mb N]"
      << " [--auto-stop-first-code-run N] [--auto-stop-min-steps N]"
      << " [--tail-stop-repeat-frames N] [--tail-stop-min-steps N]"
      << " [--trim-tail-repeat-min N] [--trim-tail-keep N] [--eos-min-steps N]"
      << " [--do-sample] [--temperature F] [--top-k N] [--sample-seed N]\n";
      << " [--lang LANG] (e.g. chinese, english, german, italian, portuguese, spanish, japanese, korean, french, russian, beijing_dialect, sichuan_dialect)\n";
}

bool ParseInt(const std::string& s, int* out) {
  const char* b = s.data();
  const char* e = s.data() + s.size();
  auto [ptr, ec] = std::from_chars(b, e, *out);
  return ec == std::errc{} && ptr == e;
}

bool ParseInt64(const std::string& s, int64_t* out) {
  const char* b = s.data();
  const char* e = s.data() + s.size();
  auto [ptr, ec] = std::from_chars(b, e, *out);
  return ec == std::errc{} && ptr == e;
}

bool ParseFloat(const std::string& s, float* out) {
  char* end = nullptr;
  errno = 0;
  const float v = std::strtof(s.c_str(), &end);
  if (errno != 0 || end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

bool ParseOrtOpt(const std::string& s, GraphOptimizationLevel* out) {
  if (s == "disable") {
    *out = GraphOptimizationLevel::ORT_DISABLE_ALL;
    return true;
  }
  if (s == "basic") {
    *out = GraphOptimizationLevel::ORT_ENABLE_BASIC;
    return true;
  }
  if (s == "extended") {
    *out = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    return true;
  }
  if (s == "all") {
    *out = GraphOptimizationLevel::ORT_ENABLE_ALL;
    return true;
  }
  return false;
}

int ParseLangStr(const std::string& s) {
    static const std::unordered_map<std::string, int> langMap = {
        {"chinese", 2055},
        {"english", 2050},
        {"german", 2053},
        {"italian", 2070},
        {"portuguese", 2071},
        {"spanish", 2054},
        {"japanese", 2058},
        {"korean", 2064},
        {"french", 2061},
        {"russian", 2069},
        {"beijing_dialect", 2074},
        {"sichuan_dialect", 2062}
    };
    
    auto it = langMap.find(s);
    return it != langMap.end() ? it->second : -1;
}

bool IsErrorPcm(const std::vector<float>& pcm, float* code) {
  if (pcm.size() == 1 && pcm[0] < 0.0f) {
    *code = pcm[0];
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  QWEN3TTS::TtsConfig cfg;
  QWEN3TTS::GenerationParams gen;
  gen.wav_out = "./output.wav";

  auto require_value = [&](int& i, const std::string& flag, std::string* out) -> bool {
    if (i + 1 >= argc) {
      std::cerr << "Error: missing value for " << flag << "\n";
      return false;
    }
    *out = argv[++i];
    return true;
  };

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    std::string value;

    if (flag == "--onnx-dir") {
      if (!require_value(i, flag, &cfg.model.path)) return 2;
    } else if (flag == "--text") {
      if (!require_value(i, flag, &gen.text)) return 2;
    } else if (flag == "--instruct") {
      if (!require_value(i, flag, &gen.instruct)) return 2;
    } else if (flag == "--output-wav") {
      if (!require_value(i, flag, &gen.wav_out)) return 2;
    } 
    else if (flag == "--save-codes-file") {
      if (!require_value(i, flag, &gen.codes_out)) return 2;
    } else if (flag == "--max-steps") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.max_steps = v;
    } else if (flag == "--prefill-builder-file") {
      if (!require_value(i, flag, &cfg.model.prefill_builder_file)) return 2;
    } else if (flag == "--talker-prefill-file") {
      if (!require_value(i, flag, &cfg.model.talker_prefill_file)) return 2;
    } else if (flag == "--talker-decode-file") {
      if (!require_value(i, flag, &cfg.model.talker_decode_file)) return 2;
    } else if (flag == "--speech-tokenizer-file") {
      if (!require_value(i, flag, &cfg.model.speech_tokenizer_file)) return 2;
    } else if (flag == "--cp-dynamic-file") {
      if (!require_value(i, flag, &cfg.model.cp_dynamic_file)) return 2;
    } else if (flag == "--cp-step-pattern") {
      if (!require_value(i, flag, &cfg.model.cp_step_pattern)) return 2;
    } else if (flag == "--tokenizer-vocab-file") {
      if (!require_value(i, flag, &cfg.model.vocab_file)) return 2;
    } else if (flag == "--tokenizer-merges-file") {
      if (!require_value(i, flag, &cfg.model.merges_file)) return 2;
    } else if (flag == "--tokenizer-config-file") {
      if (!require_value(i, flag, &cfg.model.tokenizer_config_file)) return 2;
    } else if (flag == "--ort-opt") {
      if (!require_value(i, flag, &value) || !ParseOrtOpt(value, &cfg.ort_opt)) {
        std::cerr << "Error: invalid --ort-opt value: " << value << " (use disable|basic|extended|all)\n";
        return 2;
      }
    } else if (flag == "--lang") {
      if (!require_value(i, flag, &value)) return 2;
      int lang_id = ParseLangStr(value);
      if (lang_id < 0) {
        std::cerr << "Error: invalid language: " << value << "\n";
        return 2;
      }
      gen.codec_lang = std::vector<long>{static_cast<long>(lang_id)};
    } else if (flag == "--intra-threads") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      cfg.intra_threads = v;
    } else if (flag == "--inter-threads") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      cfg.inter_threads = v;
    } else if (flag == "--device") {
      if (!require_value(i, flag, &cfg.device)) return 2;
    } else if (flag == "--prefill-device") {
      if (!require_value(i, flag, &cfg.prefill_device)) return 2;
    } else if (flag == "--talker-device") {
      if (!require_value(i, flag, &cfg.talker_device)) return 2;
    } else if (flag == "--cp-device") {
      if (!require_value(i, flag, &cfg.cp_device)) return 2;
    } else if (flag == "--vocoder-device") {
      if (!require_value(i, flag, &cfg.vocoder_device)) return 2;
    } else if (flag == "--gpu-device-id") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      cfg.gpu_device_id = v;
    } else if (flag == "--gpu-mem-limit-mb") {
      int64_t v = 0;
      if (!require_value(i, flag, &value) || !ParseInt64(value, &v)) {
        std::cerr << "Error: invalid int64 for " << flag << ": " << value << "\n";
        return 2;
      }
      cfg.gpu_mem_limit_mb = v;
    } else if (flag == "--auto-stop-first-code-run") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.auto_stop_first_code_run = v;
    } else if (flag == "--auto-stop-min-steps") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.auto_stop_min_steps = v;
    } else if (flag == "--tail-stop-repeat-frames") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.tail_stop_repeat_frames = v;
    } else if (flag == "--tail-stop-min-steps") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.tail_stop_min_steps = v;
    } else if (flag == "--trim-tail-repeat-min") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.trim_tail_repeat_min = v;
    } else if (flag == "--trim-tail-keep") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.trim_tail_keep = v;
    } else if (flag == "--eos-min-steps") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.eos_min_steps = v;
    } else if (flag == "--do-sample") {
      gen.do_sample = true;
    } else if (flag == "--temperature") {
      float v = 0.0f;
      if (!require_value(i, flag, &value) || !ParseFloat(value, &v)) {
        std::cerr << "Error: invalid float for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.temperature = v;
    } else if (flag == "--top-k") {
      int v = 0;
      if (!require_value(i, flag, &value) || !ParseInt(value, &v)) {
        std::cerr << "Error: invalid int for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.top_k = v;
    } else if (flag == "--sample-seed") {
      int64_t v = 0;
      if (!require_value(i, flag, &value) || !ParseInt64(value, &v)) {
        std::cerr << "Error: invalid int64 for " << flag << ": " << value << "\n";
        return 2;
      }
      gen.seed = v;
    } else {
      std::cerr << "Error: unknown flag: " << flag << "\n";
      return 2;
    }
  }

  if (cfg.model.path.empty()) {
    std::cerr << "Error: --onnx-dir is required\n";
    return 2;
  }
  if (gen.text.empty()) {
    std::cerr << "Error: --text is required\n";
    return 2;
  }
  if (gen.instruct.empty()) {
    std::cerr << "Error: --instruct is required\n";
    return 2;
  }

  std::filesystem::path out_path(gen.wav_out);
  if (out_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(out_path.parent_path(), ec);
    if (ec) {
      std::cerr << "Error: cannot create output directory: " << ec.message() << "\n";
      return 2;
    }
  }

  QWEN3TTS::Voice* voice = new QWEN3TTS::Voice();
  if (!voice->load(cfg)) {
    std::cerr << "Load failed with error code: " << voice->lastErrorCode()
              << " (" << voice->lastErrorMessage() << ")\n";
    delete voice;
    return 3;
  }
  std::vector<float> pcm = voice->generateVoice(gen);
  voice->unload();
  delete voice;

  float err_code = 0.0f;
  if (IsErrorPcm(pcm, &err_code)) {
    std::cerr << "Generation failed with error code: " << static_cast<int>(err_code) << "\n";
    return 3;
  }

  std::string wav_err;
  if (!QWEN3TTSUTILS::WriteWavPcm16Safe(gen.wav_out, pcm, 24000, &wav_err)) {
    std::cerr << "WAV write failed: " << wav_err << "\n";
    return 4;
  }
  std::cout << "Saved wav: " << gen.wav_out << "\n";
  return 0;
}
