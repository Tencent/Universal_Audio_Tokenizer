import glob
import os
import numpy as np
import safetensors
import torch
import torchaudio
from transformers import WhisperFeatureExtractor
from .configuration_whisper import WhisperVQConfig
from .flow_inference import AudioDecoder
from .modeling_whisper import WhisperVQEncoder

# Global cache for resampling transforms
# Key: source sample rate (int) -> Value: cached Resample transform object
_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def load_quantize_encoder(model_path):
    """
    Load the quantized encoder from the given model path.
    This function extracts only the encoder parameters from the checkpoint files.
    """
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


def save_encoder(model_path, output_path):
    """
    Save the quantized encoder to the specified output path.
    This function extracts the encoder parameters and saves them in HuggingFace format.
    """
    os.makedirs(output_path, exist_ok=True)

    encoder = load_quantize_encoder(model_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

    # Check if the model is wrapped in DataParallel or DistributedDataParallel and get the underlying model
    if hasattr(encoder, 'module'):
        # wrapped in DataParallel or DistributedDataParallel
        model_to_save = encoder.module
    else:
        # not wrapped, save the model directly
        model_to_save = encoder

    # Save the model and feature extractor in HuggingFace format with safe serialization
    model_to_save.save_pretrained(output_path, safe_serialization=True)
    feature_extractor.save_pretrained(output_path)


def extract_audio_token(
    model: WhisperVQEncoder,
    feature_extractor: WhisperFeatureExtractor,
    audios,
    batch_size=128,
    device='cuda',
):
    """
    Extract audio tokens using Universal Audio Tokenizer.

    Args:
        model (WhisperVQEncoder): Pre-trained Universal Audio Tokenizer encoder model
        feature_extractor (WhisperFeatureExtractor): Whisper feature extractor for preprocessing
        audios: List of audio file paths or (wav, sample_rate) tuples
        batch_size (int): Batch size for processing audio segments
        device (str): Device to run inference on ('cuda' or 'cpu')

    Returns:
        list: List of token sequences, one for each input audio file
    """
    with torch.no_grad():
        wavs, indices = [], []  # wavs: list of np.arrays, indices: list of ints
        for idx, audio in enumerate(audios):
            if isinstance(audio, tuple):
                wav, sample_rate = audio
                # Convert numpy array to torch tensor if needed
                if isinstance(wav, torch.Tensor):
                    wav = wav.to(device)
                elif isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav).unsqueeze(0).to(device)
                elif isinstance(wav, list):
                    wav = torch.tensor(wav).unsqueeze(0).to(device)
                else:
                    raise ValueError("Unsupported audio format in tuple. Must be np.ndarray, torch.Tensor, or list.")
            else:
                wav, sample_rate = torchaudio.load(audio)
                wav = wav.to(device)

            # Resample the audio to 16kHz if needed
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to(device)
                wav = _resample_buffer[sample_rate](wav)

            # Convert to mono by taking the first channel
            wav = wav[0]
            wav = wav.cpu().numpy()

            # Split the audio into 30-second segments to handle long files
            time_step = 0
            while time_step * 16000 < wav.shape[0]:
                wav_segment = wav[time_step * 16000: (time_step + 30) * 16000]
                wavs.append(wav_segment)
                indices.append(idx)     # Track the original audio index for each segment
                time_step += 30

        # Determine the stride for feature extraction based on model config
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_audio_tokens = [[] for _ in range(len(audios))]

        # Process audio segments in batches
        for start in range(0, len(wavs), batch_size):
            features = feature_extractor(wavs[start: start + batch_size], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device=device,
                                         padding="longest", pad_to_multiple_of=stride)
            features = features.to(device=device)
            outputs = model(**features)
            audio_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
            assert attention_mask.shape == audio_tokens.shape
            for i in range(len(audio_tokens)):
                idx = indices[start + i]    # Get original audio index
                audio_token = audio_tokens[i][attention_mask[i].bool()].tolist()    # Filter valid tokens
                all_audio_tokens[idx].extend(audio_token)   # Append to token sequence

        return all_audio_tokens


def speech_token_to_wav(audio_decoder: AudioDecoder, token_ids, embedding=None):
    """
    Convert speech tokens back to audio waveform using the audio decoder.

    Args:
        audio_decoder (AudioDecoder): audio decoder model
        token_ids: List or tensor of speech token IDs to decode
        embedding (torch.Tensor, optional): Speaker/style embedding for voice conditioning.
                                          If None, uses zero embedding (neutral voice)
        output_path (str, optional): Path to save the generated audio file.
                                   If None, audio is not saved to disk

    Returns:
        tuple: (waveform_tensor, sampling_rate)
            - waveform_tensor: Generated audio as torch.Tensor
            - sampling_rate: Audio sampling rate (typically 22050Hz)
    """
    if embedding is None:
        embedding = torch.zeros(1, 192)

    with torch.amp.autocast('cuda'):
        tts_speech, sampling_rate = audio_decoder.offline_inference(
            torch.tensor(token_ids).unsqueeze(0),
            embedding=embedding,
        )
    torch.cuda.empty_cache()

    return tts_speech, sampling_rate
