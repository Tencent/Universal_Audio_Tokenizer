import argparse
import os
import sys
import torch
import torchaudio
from transformers import WhisperFeatureExtractor
from src.model.modeling_whisper import WhisperVQEncoder
from src.model.flow_inference import AudioDecoder
from src.model.utils import extract_audio_token, speech_token_to_wav


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize and reconstruct audio using Universal Audio Tokenizer")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for inference (auto, cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory (including the \
              tokenizer subdirectory and the decoder subdirectory)"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to the audio file(s)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 34, "Arguments", "=" * 35)
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Audio path: {args.audio_path}")
    print("-" * 80)

    # Load tokenier model and feature extractor
    tokenizer_path = os.path.join(args.model_path, "tokenizer")
    tokenizer = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    # Set up system path for third-party dependencies
    sys.path.insert(0, "./src")
    sys.path.insert(0, "./third_party/Matcha-TTS")
    # Load audio decoder
    decoder_path = os.path.join(args.model_path, "decoder")
    audio_decoder = AudioDecoder(
        config_path=os.path.join(decoder_path, "config.yaml"),
        flow_ckpt_path=os.path.join(decoder_path, "flow.pt"),
        hift_ckpt_path=os.path.join(decoder_path, "hift.pt"),
        device=device
    )

    # List of input audio files
    audios = args.audio_path

    # Extract audio tokens
    print("")
    print("=" * 33, "Tokenization", "=" * 33)
    all_tokens = extract_audio_token(tokenizer, feature_extractor, audios, device=device)
    for i, (audio, token_ids) in enumerate(zip(audios, all_tokens)):
        print(f"[{i+1}/{len(audios)}] `{os.path.basename(audio)}` Generated {len(token_ids)} tokens:\n{token_ids}")
        print("-" * 80)

    # Reconstruct audio from tokens
    print("")
    print("=" * 32, "Reconstruction", "=" * 32)
    for i, (audio, token_ids) in enumerate(zip(audios, all_tokens)):
        # Get reconstructed audio waveform and sampling rate
        tts_speech, sampling_rate = speech_token_to_wav(audio_decoder, token_ids)
        # Save audio to output_path
        output_path = os.path.join("reconstruction", os.path.basename(audio))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)    # Ensure directory exists
        torchaudio.save(output_path, tts_speech, sampling_rate)
        print(f"[{i+1}/{len(audios)}] Reconstructed audio saved to: `{output_path}`")
    print("-" * 80)
