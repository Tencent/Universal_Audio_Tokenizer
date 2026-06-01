import json
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from src.train.wer import text2tokens, compute_one_wer_info, WerStats
from src.train.sap import (
    is_valid_SAP,
    caption2json,
    get_average_llm_judge_score,
    get_voted_llm_judge_response
)


def evaluate_asr(model, eval_dataloader, tokenizer, device, max_length=300, rank=0, output_file=None):
    """Evaluate model and compute WER."""
    model.eval()

    output_list = []

    wer_stat = WerStats()

    with torch.no_grad():
        for eval_batch in tqdm(eval_dataloader, desc="Evaluating", disable=rank != 0):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            input_features = eval_batch["input_features"]
            attention_mask = eval_batch["attention_mask"]

            # Get generated tokens from the model
            generate_model = model.module if hasattr(model, 'module') else model
            generated_tokens = generate_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                max_length=max_length
            )

            # Decode generated tokens and reference labels to strings
            pred_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(eval_batch["labels"], skip_special_tokens=True)

            # Compute WER for each sample in the batch and accumulate statistics
            for pred, label in zip(pred_str, label_str):
                pred_tokens = text2tokens(pred)
                label_tokens = text2tokens(label)

                if len(label_tokens) == 0:
                    continue

                wer_info = compute_one_wer_info(label_tokens, pred_tokens)
                wer_stat.add(wer_info)

                if output_file:
                    output_list.append({
                        'pred': " ".join(pred_tokens),
                        'label': " ".join(label_tokens),
                        'wer_info': repr(wer_info)
                    })

            if rank == 0:
                print(f'rank: {rank} | Current WER info:')
                wer_stat.print()

    # Write to output file (executed only in the main process)
    if output_file:
        with open(output_file, 'w', encoding='UTF-8') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)


    # Aggregate statistics across processes
    total_ref_tensor = torch.tensor(wer_stat.total_ref(), device=device)
    total_sub_tensor = torch.tensor(wer_stat.total_sub(), device=device)
    total_del_tensor = torch.tensor(wer_stat.total_del(), device=device)
    total_ins_tensor = torch.tensor(wer_stat.total_ins(), device=device)
    dist.all_reduce(total_ref_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_sub_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_del_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_ins_tensor, op=dist.ReduceOp.SUM)

    global_avg_wer = ((total_sub_tensor + total_del_tensor + total_ins_tensor) / total_ref_tensor).item()

    if rank == 0:
        print(
            f"Global total ref {total_ref_tensor.item():6d} "
            f"sub {total_sub_tensor.item():6d} "
            f"del {total_del_tensor.item():6d} "
            f"ins {total_ins_tensor.item():6d}\n"
            f"Global WER {100.0 * global_avg_wer:6.2f}%"
        )

    return global_avg_wer


def evaluate_ser(model, eval_dataloader, tokenizer, device, max_length=300, rank=0, output_file=None):
    """Evaluate model and compute accuracy for emotion recognition tasks."""
    model.eval()

    output_list = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for eval_batch in tqdm(eval_dataloader, desc="Evaluating Emotion", disable=rank != 0):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            input_features = eval_batch["input_features"]
            attention_mask = eval_batch["attention_mask"]

            generate_model = model.module if hasattr(model, 'module') else model
            emotion_special_token_id = tokenizer.convert_tokens_to_ids("<|emotion|>")

            # Use decoder_input_ids to force the first token to be the emotion token
            batch_size = input_features.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                emotion_special_token_id,
                dtype=torch.long,
                device=device
            )

            generated_tokens = generate_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                do_sample=False,  # Use greedy decoding to ensure consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            pred_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(eval_batch["labels"], skip_special_tokens=True)

            for pred, label in zip(pred_str, label_str):
                # Extract emotion label
                pred_emotion = pred.strip().lower()
                label_emotion = label.strip().lower()

                # Calculate whether the prediction is correct
                is_correct = (pred_emotion == label_emotion)
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1

                if output_file:
                    output_list.append({
                        'pred_emotion': pred_emotion,
                        'label_emotion': label_emotion,
                        'is_correct': is_correct
                    })

    # Write to output file (executed only in the main process)
    if output_file and rank == 0:
        with open(output_file, 'w', encoding='UTF-8') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    # Aggregate statistics across processes
    correct_tensor = torch.tensor(correct_predictions, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total_predictions, dtype=torch.long, device=device)

    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    # Calculate global accuracy
    global_accuracy = (
        correct_tensor.float() / total_tensor.float()
        if total_tensor > 0
        else torch.tensor(0.0, device=device)
    )
    global_accuracy = global_accuracy.item()

    if rank == 0:
        print(f"Emotion Recognition Results:")
        print(f"  Total samples: {total_tensor.item()}")
        print(f"  Correct predictions: {correct_tensor.item()}")
        print(f"  Global Accuracy: {100.0 * global_accuracy:.2f}%")

    return global_accuracy


def evaluate_cap(model, eval_dataloader, tokenizer, device, max_length=1024, rank=0, output_file=None):
    """Evaluate model on caption task."""
    model.eval()

    output_list = []
    total_count = 0
    valid_json_count = 0
    age_correct = 0
    gender_correct = 0
    emotion_correct = 0
    accent_correct = 0
    prosody_correct = 0
    timbre_correct = 0

    wer_stat = WerStats()

    with torch.no_grad():
        for eval_batch in tqdm(eval_dataloader, desc="Evaluating Caption", disable=rank != 0):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            input_features = eval_batch["input_features"]
            attention_mask = eval_batch["attention_mask"]

            generate_model = model.module if hasattr(model, 'module') else model

            # Use <|startofcaption|> token as the decoder starting token
            caption_special_token_id = tokenizer.convert_tokens_to_ids("<|startofcaption|>")
            batch_size = input_features.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                caption_special_token_id,
                dtype=torch.long,
                device=device
            )

            generated_tokens = generate_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                do_sample=False,  # Use greedy decoding to ensure consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            pred_caption = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_caption = tokenizer.batch_decode(eval_batch["labels"], skip_special_tokens=True)

            for pred, label in zip(pred_caption, label_caption):
                total_count += 1

                sample_result = {
                    'pred_caption': pred,
                    'label_caption': label
                }

                # Overall consistency score
                avg_score, scores = get_average_llm_judge_score(
                    "prompts/caption_overall_score_prompt.txt",
                    pred,
                    label,
                    num_votes=3,
                    temperature=0.2,
                    max_tokens=50,
                )
                sample_result['overall_consistency_score'] = avg_score
                sample_result['overall_consistency_score_votes'] = scores

                # Caption2json
                pred_json_str = caption2json("prompts/caption2json_prompt.txt", pred)
                label_json_str = caption2json("prompts/caption2json_prompt.txt", label)
                is_pred_valid_json, pred_json = is_valid_SAP(pred_json_str)
                is_label_valid_json, label_json = is_valid_SAP(label_json_str)
                sample_result['pred_json'] = pred_json_str
                sample_result['label_json'] = label_json_str
                sample_result['is_pred_valid_json'] = is_pred_valid_json
                sample_result['is_label_valid_json'] = is_label_valid_json

                if is_pred_valid_json and is_label_valid_json:
                    valid_json_count += 1
                    # Transcription WER
                    pred_transcription = pred_json['transcription']
                    label_transcription = label_json['transcription']
                    if pred_transcription is not None and label_transcription is not None:
                        pred_tokens = text2tokens(pred_transcription)
                        label_tokens = text2tokens(label_transcription)
                        if len(label_tokens) > 0:
                            wer_info = compute_one_wer_info(label_tokens, pred_tokens)
                            wer_stat.add(wer_info)
                            sample_result.update({
                                'pred_transcription': " ".join(pred_tokens),
                                'label_transcription': " ".join(label_tokens),
                                'wer_info': repr(wer_info)
                            })

                    # Paralinguistics evaluations
                    # Age Accuracy
                    pred_age = pred_json['paralinguistics']['age']
                    label_age = label_json['paralinguistics']['age']
                    if pred_age is not None and label_age is not None:
                        pred_age = pred_age.strip().lower()
                        label_age = label_age.strip().lower()
                        if pred_age == label_age and label_age != '':
                            age_correct += 1
                        sample_result['age_correct'] = (pred_age == label_age)
                    elif pred_age is None and label_age is None:
                        age_correct += 1

                    # Gender Accuracy
                    pred_gender = pred_json['paralinguistics']['gender']
                    label_gender = label_json['paralinguistics']['gender']
                    if pred_gender is not None and label_gender is not None:
                        pred_gender = pred_gender.strip().lower()
                        label_gender = label_gender.strip().lower()
                        if pred_gender == label_gender and label_gender != '':
                            gender_correct += 1
                        sample_result['gender_correct'] = (pred_gender == label_gender)
                    elif pred_gender is None and label_gender is None:
                        gender_correct += 1

                    # Emotion Accuracy
                    pred_emotion = pred_json['paralinguistics']['emotion']
                    label_emotion = label_json['paralinguistics']['emotion']
                    if pred_emotion is not None and label_emotion is not None:
                        pred_emotion = pred_emotion.strip().lower()
                        label_emotion = label_emotion.strip().lower()
                        if pred_emotion == label_emotion and label_emotion != '':
                            emotion_correct += 1
                        sample_result['emotion_correct'] = (pred_emotion == label_emotion)
                    elif pred_emotion is None and label_emotion is None:
                        emotion_correct += 1

                    # Accent consistency - generate 3 times and take majority vote
                    pred_accent = pred_json['paralinguistics']['accent']
                    label_accent = label_json['paralinguistics']['accent']
                    if pred_accent is not None and label_accent is not None:
                        accent_consistency, accent_confidence = get_voted_llm_judge_response(
                            "prompts/accent_eval_prompt.txt",
                            pred_accent.strip().lower(),
                            label_accent.strip().lower(),
                            num_votes=3,
                            temperature=0.2,
                            max_tokens=50,
                        )
                        sample_result['accent_consistency'] = accent_consistency
                        sample_result['accent_confidence'] = accent_confidence
                        if accent_consistency == 'yes':
                            accent_correct += 1
                    elif pred_accent is None and label_accent is None:
                        accent_correct += 1

                    # Prosody consistency - generate 3 times and take majority vote
                    pred_prosody = pred_json['paralinguistics']['prosody']
                    label_prosody = label_json['paralinguistics']['prosody']
                    if pred_prosody is not None and label_prosody is not None:
                        prosody_consistency, prosody_confidence = get_voted_llm_judge_response(
                            "prompts/prosody_eval_prompt.txt",
                            pred_prosody.strip().lower(),
                            label_prosody.strip().lower(),
                            num_votes=3,
                            temperature=0.2,
                            max_tokens=50,
                        )
                        sample_result['prosody_consistency'] = prosody_consistency
                        sample_result['prosody_confidence'] = prosody_confidence
                        if prosody_consistency == 'yes':
                            prosody_correct += 1
                    elif pred_prosody is None and label_prosody is None:
                        prosody_correct += 1

                    # Timbre consistency - generate 3 times and take majority vote
                    pred_timbre = pred_json['paralinguistics']['timbre']
                    label_timbre = label_json['paralinguistics']['timbre']
                    if pred_timbre is not None and label_timbre is not None:
                        timbre_consistency, timbre_confidence = get_voted_llm_judge_response(
                            "prompts/timbre_eval_prompt.txt",
                            pred_timbre.strip().lower(),
                            label_timbre.strip().lower(),
                            num_votes=3,
                            temperature=0.2,
                            max_tokens=50,
                        )
                        sample_result['timbre_consistency'] = timbre_consistency
                        sample_result['timbre_confidence'] = timbre_confidence
                        if timbre_consistency == 'yes':
                            timbre_correct += 1
                    elif pred_timbre is None and label_timbre is None:
                        timbre_correct += 1

                output_list.append(sample_result)

    print(
        f"========== [{time.ctime()}] Rank {rank} finished evaluation. "
        f"Total samples: {total_count}, Valid JSON samples: {valid_json_count} ==========",
        flush=True
    )

    # Output detailed results to JSON file (only rank 0)
    if output_file and rank == 0:
        with open(output_file, 'w', encoding='UTF-8') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    # Aggregate statistics across processes
    print(f"========== [{time.ctime()}] Rank {rank} Converting scores to tensor... ==========", flush=True)
    all_valid_overall_score = [sample_result['overall_consistency_score'] for sample_result in output_list \
                               if 'overall_consistency_score' in sample_result and \
                                  sample_result['overall_consistency_score'] is not None]
    overall_score_count = len(all_valid_overall_score)
    overall_score_sum = sum(all_valid_overall_score)

    overall_score_sum_tensor = torch.tensor(overall_score_sum, dtype=torch.float, device=device)
    overall_score_count_tensor = torch.tensor(overall_score_count, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total_count, dtype=torch.long, device=device)
    valid_json_tensor = torch.tensor(valid_json_count, dtype=torch.long, device=device)
    total_ref_tensor = torch.tensor(wer_stat.total_ref(), device=device)
    total_sub_tensor = torch.tensor(wer_stat.total_sub(), device=device)
    total_del_tensor = torch.tensor(wer_stat.total_del(), device=device)
    total_ins_tensor = torch.tensor(wer_stat.total_ins(), device=device)
    age_correct_tensor = torch.tensor(age_correct, dtype=torch.long, device=device)
    gender_correct_tensor = torch.tensor(gender_correct, dtype=torch.long, device=device)
    emotion_correct_tensor = torch.tensor(emotion_correct, dtype=torch.long, device=device)
    accent_correct_tensor = torch.tensor(accent_correct, dtype=torch.long, device=device)
    prosody_correct_tensor = torch.tensor(prosody_correct, dtype=torch.long, device=device)
    timbre_correct_tensor = torch.tensor(timbre_correct, dtype=torch.long, device=device)
    dist.all_reduce(overall_score_sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(overall_score_count_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(valid_json_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_ref_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_sub_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_del_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_ins_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(age_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(gender_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(emotion_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(accent_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(prosody_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(timbre_correct_tensor, op=dist.ReduceOp.SUM)

    json_valid_rate = valid_json_tensor.item() / total_tensor.item() if total_tensor > 0 else 0.0
    wer = ((total_sub_tensor.item() + total_del_tensor.item() + total_ins_tensor.item()) / total_ref_tensor.item()) \
          if total_ref_tensor.item() > 0 else 1.0
    age_accuracy = age_correct_tensor.item() / valid_json_tensor.item() if valid_json_tensor > 0 else 0.0
    gender_accuracy = gender_correct_tensor.item() / valid_json_tensor.item() if valid_json_tensor > 0 else 0.0
    emotion_accuracy = emotion_correct_tensor.item() / valid_json_tensor.item() if valid_json_tensor > 0 else 0.0
    accent_accuracy = accent_correct_tensor.item() / valid_json_tensor.item() if valid_json_tensor > 0 else 0.0
    prosody_accuracy = prosody_correct_tensor.item() / valid_json_tensor.item() if valid_json_tensor > 0 else 0.0
    timbre_accuracy = timbre_correct_tensor.item() / valid_json_tensor.item() if valid_json_tensor > 0 else 0.0

    results = {
        'total_samples': total_tensor.item(),
        'overall_consistency_score': overall_score_sum_tensor.item() / overall_score_count_tensor.item() \
            if overall_score_count_tensor.item() > 0 else 0.0,
        'valid_json_samples': valid_json_tensor.item(),
        'valid_json_rate': json_valid_rate,
        'wer_stats': {
            'ref': total_ref_tensor.item(),
            'sub': total_sub_tensor.item(),
            'del': total_del_tensor.item(),
            'ins': total_ins_tensor.item(),
            'wer': wer
        },
        'age_correct': age_correct_tensor.item(),
        'age_accuracy': age_accuracy,
        'gender_correct': gender_correct_tensor.item(),
        'gender_accuracy': gender_accuracy,
        'emotion_correct': emotion_correct_tensor.item(),
        'emotion_accuracy': emotion_accuracy,
        'accent_correct': accent_correct_tensor.item(),
        'accent_accuracy': accent_accuracy,
        'prosody_correct': prosody_correct_tensor.item(),
        'prosody_accuracy': prosody_accuracy,
        'timbre_correct': timbre_correct_tensor.item(),
        'timbre_accuracy': timbre_accuracy
    }

    if rank == 0:
        print(f"Caption Generation Results:")
        print(f"  Total samples: {total_tensor.item()}")
        print(f"  Overall Consistency Score: {results['overall_consistency_score']:.4f}")
        print(f"  Valid JSON samples: {results['valid_json_samples']}/{results['total_samples']}",
              f"({100.0 * results['valid_json_rate']:.2f}%)")
        print(f"  WER Stats: WER {100.0 * results['wer_stats']['wer']:.2f}%",
              f"Ref {results['wer_stats']['ref']:6d} Sub {results['wer_stats']['sub']:6d}",
              f"Del {results['wer_stats']['del']:6d} Ins {results['wer_stats']['ins']:6d}")
        print(f"  Age Accuracy: {results['age_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['age_accuracy']:.2f}%)")
        print(f"  Gender Accuracy: {results['gender_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['gender_accuracy']:.2f}%)")
        print(f"  Emotion Accuracy: {results['emotion_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['emotion_accuracy']:.2f}%)")
        print(f"  Accent Accuracy: {results['accent_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['accent_accuracy']:.2f}%)")
        print(f"  Prosody Accuracy: {results['prosody_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['prosody_accuracy']:.2f}%)")
        print(f"  Timbre Accuracy: {results['timbre_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['timbre_accuracy']:.2f}%)")

    return results


def evaluate_cap_only_emotion(model, eval_dataloader, tokenizer, device, max_length=1024, rank=0, output_file=None):
    """Evaluate model on caption task."""
    model.eval()

    output_list = []
    total_count = 0
    valid_json_count = 0
    emotion_correct = 0

    with torch.no_grad():
        for eval_batch in tqdm(eval_dataloader, desc="Evaluating Caption", disable=rank != 0):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            input_features = eval_batch["input_features"]
            attention_mask = eval_batch["attention_mask"]

            generate_model = model.module if hasattr(model, 'module') else model

            # Use <|startofcaption|> token as the decoder starting token
            caption_special_token_id = tokenizer.convert_tokens_to_ids("<|startofcaption|>")
            batch_size = input_features.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                caption_special_token_id,
                dtype=torch.long,
                device=device
            )

            generated_tokens = generate_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                do_sample=False,  # Use greedy decoding to ensure consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            pred_caption = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_emotions = tokenizer.batch_decode(eval_batch["labels"], skip_special_tokens=True)

            for pred, label_emotion in zip(pred_caption, label_emotions):
                total_count += 1

                sample_result = {
                    'pred_caption': pred,
                    'label_emotion': label_emotion
                }

                # Caption2json
                pred_json_str = caption2json("prompts/caption2json_prompt.txt", pred)
                is_pred_valid_json, pred_json = is_valid_SAP(pred_json_str)
                sample_result['pred_json'] = pred_json_str
                sample_result['is_pred_valid_json'] = is_pred_valid_json

                if is_pred_valid_json:
                    valid_json_count += 1

                    # Paralinguistics evaluations
                    # Emotion Accuracy
                    pred_emotion = pred_json['paralinguistics']['emotion']
                    if pred_emotion is not None and label_emotion is not None:
                        pred_emotion = pred_emotion.strip().lower()
                        label_emotion = label_emotion.strip().lower()
                        if pred_emotion == label_emotion and label_emotion != '':
                            emotion_correct += 1
                        sample_result['emotion_correct'] = (pred_emotion == label_emotion)
                    elif pred_emotion is None and label_emotion is None:
                        emotion_correct += 1

                output_list.append(sample_result)

    # Write to output file (executed only in the main process)
    if output_file and rank == 0:
        with open(output_file, 'w', encoding='UTF-8') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    # Aggregate statistics across processes
    total_tensor = torch.tensor(total_count, dtype=torch.long, device=device)
    valid_json_tensor = torch.tensor(valid_json_count, dtype=torch.long, device=device)
    emotion_correct_tensor = torch.tensor(emotion_correct, dtype=torch.long, device=device)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(valid_json_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(emotion_correct_tensor, op=dist.ReduceOp.SUM)

    json_valid_rate = valid_json_tensor.item() / total_tensor.item() if total_tensor > 0 else 0.0
    emotion_accuracy = emotion_correct_tensor.item() / valid_json_tensor.item() if valid_json_tensor > 0 else 0.0

    results = {
        'total_samples': total_tensor.item(),
        'valid_json_samples': valid_json_tensor.item(),
        'valid_json_rate': json_valid_rate,
        'emotion_correct': emotion_correct_tensor.item(),
        'emotion_accuracy': emotion_accuracy,
    }

    if rank == 0:
        print(f"Caption Generation Results:")
        print(f"  Total samples: {total_tensor.item()}")
        print(f"  Valid JSON samples: {results['valid_json_samples']}/{results['total_samples']}",
              f"({100.0 * results['valid_json_rate']:.2f}%)")
        print(f"  Emotion Accuracy: {results['emotion_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['emotion_accuracy']:.2f}%)")

    return results


def evaluate_sap(model, eval_dataloader, tokenizer, device, max_length=1024, rank=0, output_file=None):
    """Evaluate model on SAP task."""
    model.eval()

    output_list = []

    total_count = 0
    valid_count = 0

    wer_stat = WerStats()
    age_correct = 0
    gender_correct = 0
    emotion_correct = 0
    accent_correct = 0
    prosody_correct = 0
    timbre_correct = 0

    with torch.no_grad():
        for eval_batch in tqdm(eval_dataloader, desc="Evaluating SAP", disable=rank != 0):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            input_features = eval_batch["input_features"]
            attention_mask = eval_batch["attention_mask"]

            generate_model = model.module if hasattr(model, 'module') else model

            # Use <|startofsap|> token as the decoder starting token
            sap_special_token_id = tokenizer.convert_tokens_to_ids("<|startofsap|>")

            batch_size = input_features.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                sap_special_token_id,
                dtype=torch.long,
                device=device
            )

            generated_tokens = generate_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                do_sample=False,  # Use greedy decoding to ensure consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                suppress_tokens=[],  # Do not suppress any token, allowing generation of all tokens
                begin_suppress_tokens=[],  # Do not suppress any token at the beginning either
            )

            pred_json = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_json = tokenizer.batch_decode(eval_batch["labels"], skip_special_tokens=True)

            for pred, label in zip(pred_json, label_json):
                sample_result = {
                    'pred_json': pred,
                    'label_json': label
                }

                # Determine if it is a valid json file
                is_valid_json, pred_data = is_valid_SAP(pred)
                _, label_data = is_valid_SAP(label)

                total_count += 1
                if is_valid_json:
                    valid_count += 1

                sample_result['is_valid_json'] = is_valid_json

                if pred_data is not None and label_data is not None:
                    # 1. Calculate WER
                    pred_transcription = pred_data.get('transcription', '')
                    label_transcription = label_data.get('transcription', '')
                    if pred_transcription is not None and label_transcription is not None:
                        pred_tokens = text2tokens(pred_transcription)
                        label_tokens = text2tokens(label_transcription)
                        if len(label_tokens) > 0:
                            wer_info = compute_one_wer_info(label_tokens, pred_tokens)
                            wer_stat.add(wer_info)
                            sample_result['wer_info'] = repr(wer_info)

                    # 2. Calculate classification accuracy for age, gender, emotion, etc.
                    pred_age = pred_data['paralinguistics']['age']
                    label_age = label_data['paralinguistics']['age']
                    if pred_age is not None and label_age is not None:
                        pred_age = pred_age.strip().lower()
                        label_age = label_age.strip().lower()
                        if pred_age == label_age and label_age != '':
                            age_correct += 1
                        sample_result['age_correct'] = (pred_age == label_age)
                    elif pred_age is None and label_age is None:
                        age_correct += 1

                    pred_gender = pred_data['paralinguistics']['gender']
                    label_gender = label_data['paralinguistics']['gender']
                    if pred_gender is not None and label_gender is not None:
                        pred_gender = pred_gender.strip().lower()
                        label_gender = label_gender.strip().lower()
                        if pred_gender == label_gender and label_gender != '':
                            gender_correct += 1
                        sample_result['gender_correct'] = (pred_gender == label_gender)
                    elif pred_gender is None and label_gender is None:
                        gender_correct += 1

                    pred_emotion = pred_data['paralinguistics']['emotion']
                    label_emotion = label_data['paralinguistics']['emotion']
                    if pred_emotion is not None and label_emotion is not None:
                        pred_emotion = pred_emotion.strip().lower()
                        label_emotion = label_emotion.strip().lower()
                        if pred_emotion == label_emotion and label_emotion != '':
                            emotion_correct += 1
                        sample_result['emotion_correct'] = (pred_emotion == label_emotion)
                    elif pred_emotion is None and label_emotion is None:
                        emotion_correct += 1

                    # Accent consistency - generate 3 times and take majority vote
                    pred_accent = pred_data['paralinguistics']['accent']
                    label_accent = label_data['paralinguistics']['accent']
                    if pred_accent is not None and label_accent is not None:
                        accent_consistency, accent_confidence = get_voted_llm_judge_response(
                            "prompts/accent_eval_prompt.txt",
                            pred_accent.strip().lower(),
                            label_accent.strip().lower(),
                            num_votes=3,
                            temperature=0.2,
                            max_tokens=50,
                        )
                        sample_result['accent_consistency'] = accent_consistency
                        sample_result['accent_confidence'] = accent_confidence
                        if accent_consistency == 'yes':
                            accent_correct += 1
                    elif pred_accent is None and label_accent is None:
                        accent_correct += 1

                    # Prosody consistency - generate 3 times and take majority vote
                    pred_prosody = pred_data['paralinguistics']['prosody']
                    label_prosody = label_data['paralinguistics']['prosody']
                    if pred_prosody is not None and label_prosody is not None:
                        prosody_consistency, prosody_confidence = get_voted_llm_judge_response(
                            "prompts/prosody_eval_prompt.txt",
                            pred_prosody.strip().lower(),
                            label_prosody.strip().lower(),
                            num_votes=3,
                            temperature=0.2,
                            max_tokens=50,
                        )
                        sample_result['prosody_consistency'] = prosody_consistency
                        sample_result['prosody_confidence'] = prosody_confidence
                        if prosody_consistency == 'yes':
                            prosody_correct += 1
                    elif pred_prosody is None and label_prosody is None:
                        prosody_correct += 1

                    # Timbre consistency - generate 3 times and take majority vote
                    pred_timbre = pred_data['paralinguistics']['timbre']
                    label_timbre = label_data['paralinguistics']['timbre']
                    if pred_timbre is not None and label_timbre is not None:
                        timbre_consistency, timbre_confidence = get_voted_llm_judge_response(
                            "prompts/timbre_eval_prompt.txt",
                            pred_timbre.strip().lower(),
                            label_timbre.strip().lower(),
                            num_votes=3,
                            temperature=0.2,
                            max_tokens=50,
                        )
                        sample_result['timbre_consistency'] = timbre_consistency
                        sample_result['timbre_confidence'] = timbre_confidence
                        if timbre_consistency == 'yes':
                            timbre_correct += 1
                    elif pred_timbre is None and label_timbre is None:
                        timbre_correct += 1

                    # Non-Linguistic Events
                    pred_nl_desc = pred_data['nonLinguisticEvents']['description']
                    label_nl_desc = label_data['nonLinguisticEvents']['description']
                    avg_score, scores = get_average_llm_judge_score(
                        "prompts/non_linguistic_overall_score_prompt.txt",
                        pred_nl_desc,
                        label_nl_desc,
                        num_votes=3,
                        temperature=0.2,
                        max_tokens=50,
                    )
                    sample_result['non_linguistic_overall_consistency_score'] = avg_score
                    sample_result['non_linguistic_overall_consistency_score_votes'] = scores

                if output_file:
                    output_list.append(sample_result)


    # Write to output file (executed only in the main process)
    if output_file and rank == 0:
        with open(output_file, 'w', encoding='UTF-8') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    print(
        f"========== [{time.ctime()}] Rank {rank} finished evaluation. "
        f"Total samples: {total_count}, Valid samples: {valid_count} ==========",
        flush=True
    )

    # Aggregate statistics across processes
    # all_valid_overall_score = [sample_result['overall_consistency_score'] for sample_result in output_list \
    # if 'overall_consistency_score' in sample_result and \
    #    sample_result['overall_consistency_score'] is not None]
    # overall_score_count = len(all_valid_overall_score)
    # overall_score_sum = sum(all_valid_overall_score)

    all_valid_non_linguistic_overall_score = [
        sample_result['non_linguistic_overall_consistency_score'] for sample_result in output_list \
        if 'non_linguistic_overall_consistency_score' in sample_result and \
            sample_result['non_linguistic_overall_consistency_score'] is not None
    ]
    non_linguistic_overall_score_count = len(all_valid_non_linguistic_overall_score)
    non_linguistic_overall_score_sum = sum(all_valid_non_linguistic_overall_score)

    # overall_score_sum_tensor = torch.tensor(overall_score_sum, dtype=torch.float, device=device)
    # overall_score_count_tensor = torch.tensor(overall_score_count, dtype=torch.long, device=device)
    non_linguistic_overall_score_sum_tensor = \
        torch.tensor(non_linguistic_overall_score_sum, dtype=torch.float, device=device)
    non_linguistic_overall_score_count_tensor = \
        torch.tensor(non_linguistic_overall_score_count, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total_count, dtype=torch.long, device=device)
    valid_tensor = torch.tensor(valid_count, dtype=torch.long, device=device)
    total_ref_tensor = torch.tensor(wer_stat.total_ref(), device=device)
    total_sub_tensor = torch.tensor(wer_stat.total_sub(), device=device)
    total_del_tensor = torch.tensor(wer_stat.total_del(), device=device)
    total_ins_tensor = torch.tensor(wer_stat.total_ins(), device=device)
    age_correct_tensor = torch.tensor(age_correct, dtype=torch.long, device=device)
    gender_correct_tensor = torch.tensor(gender_correct, dtype=torch.long, device=device)
    emotion_correct_tensor = torch.tensor(emotion_correct, dtype=torch.long, device=device)
    accent_correct_tensor = torch.tensor(accent_correct, dtype=torch.long, device=device)
    prosody_correct_tensor = torch.tensor(prosody_correct, dtype=torch.long, device=device)
    timbre_correct_tensor = torch.tensor(timbre_correct, dtype=torch.long, device=device)
    # dist.all_reduce(overall_score_sum_tensor, op=dist.ReduceOp.SUM)
    # dist.all_reduce(overall_score_count_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(non_linguistic_overall_score_sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(non_linguistic_overall_score_count_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(valid_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_ref_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_sub_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_del_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_ins_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(age_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(gender_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(emotion_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(accent_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(prosody_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(timbre_correct_tensor, op=dist.ReduceOp.SUM)

    json_valid_rate = valid_tensor.item() / total_tensor.item() if total_tensor > 0 else 0.0
    wer = ((total_sub_tensor.item() + total_del_tensor.item() + total_ins_tensor.item()) / total_ref_tensor.item()) \
          if total_ref_tensor.item() > 0 else 1.0
    age_accuracy = age_correct_tensor.item() / valid_tensor.item() if valid_tensor > 0 else 0.0
    gender_accuracy = gender_correct_tensor.item() / valid_tensor.item() if valid_tensor > 0 else 0.0
    emotion_accuracy = emotion_correct_tensor.item() / valid_tensor.item() if valid_tensor > 0 else 0.0
    accent_accuracy = accent_correct_tensor.item() / valid_tensor.item() if valid_tensor > 0 else 0.0
    prosody_accuracy = prosody_correct_tensor.item() / valid_tensor.item() if valid_tensor > 0 else 0.0
    timbre_accuracy = timbre_correct_tensor.item() / valid_tensor.item() if valid_tensor > 0 else 0.0

    results = {
        'total_samples': total_tensor.item(),
        # 'overall_consistency_score': overall_score_sum_tensor.item() / overall_score_count_tensor.item() \
        #     if overall_score_count_tensor.item() > 0 else 0.0,
        'non_linguistic_overall_consistency_score': (
            non_linguistic_overall_score_sum_tensor.item() / \
            non_linguistic_overall_score_count_tensor.item()
        ) \
            if non_linguistic_overall_score_count_tensor.item() > 0 else 0.0,
        'valid_json_samples': valid_tensor.item(),
        'valid_json_rate': json_valid_rate,
        'wer_stats': {
            'ref': total_ref_tensor.item(),
            'sub': total_sub_tensor.item(),
            'del': total_del_tensor.item(),
            'ins': total_ins_tensor.item(),
            'wer': wer
        },
        'age_correct': age_correct_tensor.item(),
        'age_accuracy': age_accuracy,
        'gender_correct': gender_correct_tensor.item(),
        'gender_accuracy': gender_accuracy,
        'emotion_correct': emotion_correct_tensor.item(),
        'emotion_accuracy': emotion_accuracy,
        'accent_correct': accent_correct_tensor.item(),
        'accent_accuracy': accent_accuracy,
        'prosody_correct': prosody_correct_tensor.item(),
        'prosody_accuracy': prosody_accuracy,
        'timbre_correct': timbre_correct_tensor.item(),
        'timbre_accuracy': timbre_accuracy
    }

    if rank == 0:
        print(f"SAP Results:")
        print(f"  Total samples: {results['total_samples']}")
        # print(f"  Overall Consistency Score: {results['overall_consistency_score']:.4f}")
        print(f"  Valid JSON samples: {results['valid_json_samples']}/{results['total_samples']}",
              f"({100.0 * results['valid_json_rate']:.2f}%)")
        print(f"  WER Stats: WER {100.0 * results['wer_stats']['wer']:.2f}%",
              f"Ref {results['wer_stats']['ref']:6d} Sub {results['wer_stats']['sub']:6d}",
              f"Del {results['wer_stats']['del']:6d} Ins {results['wer_stats']['ins']:6d}")
        print(f"  Age Accuracy: {results['age_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['age_accuracy']:.2f}%)")
        print(f"  Gender Accuracy: {results['gender_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['gender_accuracy']:.2f}%)")
        print(f"  Emotion Accuracy: {results['emotion_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['emotion_accuracy']:.2f}%)")
        print(f"  Accent Accuracy: {results['accent_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['accent_accuracy']:.2f}%)")
        print(f"  Prosody Accuracy: {results['prosody_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['prosody_accuracy']:.2f}%)")
        print(f"  Timbre Accuracy: {results['timbre_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['timbre_accuracy']:.2f}%)")
        print(f"  Non-Linguistic Overall Consistency Score: {results['non_linguistic_overall_consistency_score']:.4f}")

    return results


def evaluate_sap_only_emotion(model, eval_dataloader, tokenizer, device, max_length=1024, rank=0, output_file=None):
    """Evaluate model on sap task."""
    model.eval()

    output_list = []

    total_count = 0
    valid_count = 0

    emotion_correct = 0

    with torch.no_grad():
        for eval_batch in tqdm(eval_dataloader, desc="Evaluating SAP Emotion", disable=rank != 0):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            input_features = eval_batch["input_features"]
            attention_mask = eval_batch["attention_mask"]

            generate_model = model.module if hasattr(model, 'module') else model

            # Use <|startofsap|> token as the decoder starting token
            sap_special_token_id = tokenizer.convert_tokens_to_ids("<|startofsap|>")

            batch_size = input_features.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                sap_special_token_id,
                dtype=torch.long,
                device=device
            )

            generated_tokens = generate_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                do_sample=False,  # Use greedy decoding to ensure consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                suppress_tokens=[],  # Do not suppress any token, allowing generation of all tokens
                begin_suppress_tokens=[],  # Do not suppress any token at the beginning either
            )

            pred_json = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_emotions = tokenizer.batch_decode(eval_batch["labels"], skip_special_tokens=True)

            for pred, label_emotion in zip(pred_json, label_emotions):
                sample_result = {
                    'pred_json': pred,
                    'label_emotion': label_emotion
                }

                # Determine if it is a valid json file
                is_valid_json, pred_data = is_valid_SAP(pred)

                total_count += 1
                if is_valid_json:
                    valid_count += 1

                sample_result['is_valid_json'] = is_valid_json

                if pred_data is not None:
                    pred_emotion = pred_data['paralinguistics']['emotion']
                    if pred_emotion is not None and label_emotion is not None:
                        pred_emotion = pred_emotion.strip().lower()
                        label_emotion = label_emotion.strip().lower()
                        if pred_emotion == label_emotion and label_emotion != '':
                            emotion_correct += 1
                        sample_result['emotion_correct'] = (pred_emotion == label_emotion)
                    elif pred_emotion is None and label_emotion is None:
                        emotion_correct += 1

                if output_file:
                    output_list.append(sample_result)


    # Write to output file (executed only in the main process)
    if output_file and rank == 0:
        with open(output_file, 'w', encoding='UTF-8') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    print(
        f"========== [{time.ctime()}] Rank {rank} finished evaluation. "
        f"Total samples: {total_count}, Valid samples: {valid_count} ==========",
        flush=True
    )

    # Aggregate statistics across processes
    total_tensor = torch.tensor(total_count, dtype=torch.long, device=device)
    valid_tensor = torch.tensor(valid_count, dtype=torch.long, device=device)
    emotion_correct_tensor = torch.tensor(emotion_correct, dtype=torch.long, device=device)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(valid_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(emotion_correct_tensor, op=dist.ReduceOp.SUM)

    json_valid_rate = (valid_tensor.item() / total_tensor.item()) if total_tensor > 0 else 0.0
    emotion_accuracy = (emotion_correct_tensor.item() / valid_tensor.item()) if valid_tensor > 0 else 0.0

    results = {
        'total_samples': total_tensor.item(),
        'valid_json_samples': valid_tensor.item(),
        'valid_json_rate': json_valid_rate,
        'emotion_correct': emotion_correct_tensor.item(),
        'emotion_accuracy': emotion_accuracy,
    }

    if rank == 0:
        print(f"SAP Results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Valid JSON samples: {results['valid_json_samples']}/{results['total_samples']}",
              f"({100.0 * results['valid_json_rate']:.2f}%)")
        print(f"  Emotion Accuracy: {results['emotion_correct']}/{results['valid_json_samples']}",
              f"({100.0 * results['emotion_accuracy']:.2f}%)")

    return results
