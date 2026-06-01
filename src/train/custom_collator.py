from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import librosa
import torch
import torchaudio


@dataclass
class CustomDataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any
    model_config: Any
    decoder_start_token_id: int
    padding_strategy: str = "longest"
    dataset_name: str = "default"
    min_text_length: int = 0

    def __post_init__(self):
        self.resamplers = {}  # Dictionary to store different resamplers

    def _mixed_preprocessor(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        for feature in features:
            if 'audio_data' in feature:
                feature['speech'], feature['sample_rate'] = torchaudio.load(BytesIO(feature['audio_data']))
                # Convert to numpy
                if feature['sample_rate'] != self.feature_extractor.sampling_rate:
                    resampler = self.resamplers.get(feature['sample_rate'])
                    if resampler is None:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=feature['sample_rate'],
                            new_freq=self.feature_extractor.sampling_rate
                        )
                        self.resamplers[feature['sample_rate']] = resampler
                    feature['speech'] = resampler(feature['speech'])
                    feature['sample_rate'] = self.feature_extractor.sampling_rate
                # Ensure 1D array (take mean across channels if multi-channel)
                feature['speech'] = feature['speech'].mean(dim=0).numpy()
                del feature['audio_data']

            elif 'audio' in feature:
                feature['speech'] = feature["audio"]['array']
                feature['sample_rate'] = feature["audio"]['sampling_rate']
                del feature['audio']

                if feature['sample_rate'] != self.feature_extractor.sampling_rate:
                    feature['speech'] = librosa.resample(
                        feature['speech'],
                        orig_sr=feature['sample_rate'],
                        target_sr=self.feature_extractor.sampling_rate
                    )
                    feature['sample_rate'] = self.feature_extractor.sampling_rate

            if 'sap' in feature:
                feature['task'] = 'sap'
            elif 'caption' in feature:
                feature['task'] = 'caption'
            else:
                if 'task' not in feature:
                    feature['task'] = 'asr'
                if feature['task'] == 'emotion':
                    feature['task'] = 'ser'

        return features

    def dataset_check(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> None:
        for feature in features:
            assert 'speech' in feature
            assert 'sample_rate' in feature
            assert 'task' in feature
            assert feature['task'] in ['asr', 'ser', 'caption', 'sap'], f"Unknown task type: {feature['task']}"
            if feature['task'] == 'asr':
                assert 'text' in feature
            elif feature['task'] == 'ser':
                assert 'emotion' in feature
            assert feature['sample_rate'] == self.feature_extractor.sampling_rate
            assert feature['sample_rate'] == 16000


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Process audio features on-the-fly
        features = self._mixed_preprocessor(features)

        # Process clean audio features
        self.dataset_check(features)
        input_features = []
        stride = 2 * self.model_config.pooling_kernel_size * self.feature_extractor.hop_length
        all_features = self.feature_extractor(
            [feature["speech"] for feature in features],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            padding=self.padding_strategy,
            pad_to_multiple_of=stride,
        )

        audio_features = all_features.input_features
        attention_mask = all_features.attention_mask

        for i in range(len(audio_features)):
            input_features.append({"input_features": audio_features[i], "attention_mask": attention_mask[i]})

        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = []
        start_token_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        emotion_id = self.tokenizer.convert_tokens_to_ids("<|emotion|>")
        start_caption_id = self.tokenizer.convert_tokens_to_ids("<|startofcaption|>")
        start_sap_id = self.tokenizer.convert_tokens_to_ids("<|startofsap|>")
        for feature in features:
            if feature['task'] == 'ser':
                emotion_type_id = self.tokenizer(feature['emotion']).input_ids
                label_ids = [emotion_id] + emotion_type_id[2:]  # Start token is <|emotion|>, id = 51865
            elif feature['task'] == 'asr':
                if len(feature["text"]) < self.min_text_length:
                    label_ids = [start_token_id, -100]  # remove data with short text
                    print(f"Text `{feature['text']}` is less than min_text_length {self.min_text_length}!"
                          f"set label_ids to {label_ids}")
                else:
                    label_ids = self.tokenizer(feature["text"]).input_ids
                    # Start token is <|startoftranscript|>, id = 50258
            elif feature['task'] == 'caption':
                if len(feature['caption']) < self.min_text_length:
                    label_ids = [start_caption_id, -100]    # remove data with short text
                    print(f"Caption `{feature['caption']}` is less than min_text_length {self.min_text_length}!"
                          f"set label_ids to {label_ids}")
                else:
                    caption_ids = self.tokenizer(feature["caption"]).input_ids
                    label_ids = [start_caption_id] + caption_ids[2:]
                    # Start token is <|startofcaption|>, skip start tokens <|startoftranscript|>, <|notimestamps|>
            elif feature['task'] == 'sap':
                if len(feature['sap']) < self.min_text_length:
                    label_ids = [start_sap_id, -100]    # remove data with short text
                    print(f"SAP `{feature['sap']}` is less than min_text_length {self.min_text_length}!"
                          f"set label_ids to {label_ids}")
                else:
                    sap_ids = self.tokenizer(feature['sap']).input_ids
                    label_ids = [start_sap_id] + sap_ids[2:]
                    # Start token is <|startofsap|>, skip start tokens <|startoftranscript|>, <|notimestamps|>
            else:
                raise ValueError(f"Unknown task type: {feature['task']}")

            if len(label_ids) > self.model_config.max_target_positions:
                label_ids = label_ids[:self.model_config.max_target_positions]

            label_features.append({"input_ids": label_ids})

        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        full_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        decoder_input_ids = full_labels[:, :-1]
        labels = full_labels[:, 1:]
        batch["labels"] = labels
        decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == -100, self.tokenizer.pad_token_id)
        # 50256 is the id of <|endoftext|>
        batch["decoder_input_ids"] = decoder_input_ids
        assert (decoder_input_ids < len(self.tokenizer)).all()
        return batch


def process_emotion_info(features, domain, tokenizer, decoder_start_token_id):
    domain_features = []
    for feature in features:
        domain_ids = tokenizer(feature.get(domain, "None")).input_ids
        domain_features.append({"input_ids": domain_ids})

    domains_batch = tokenizer.pad(domain_features, return_tensors="pt")
    domains = domains_batch["input_ids"].masked_fill(domains_batch.attention_mask.ne(1), -100)

    if (domains[:, 0] == decoder_start_token_id).all().cpu().item():
        domains = domains[:, 1:]

    return domains
