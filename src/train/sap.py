import json
import os
import requests

EMOTION_MAPPING = {
    "Frustration": "Sadness",
    "Anxiety": "Neutral",
    "Affectionate": "Happiness",
    "Exasperated": "Anger",
    "Exasperation": "Anger",
    "Sarcasm": "Disgust",
    "Impatience": "Anger",
    "Mocking": "Disgust",
    "Mockery": "Disgust",
    "Confusion": "Surprise",
    "Playful": "Happiness",
    "Playfulness": "Happiness",
    "Excitement": "Happiness",
    "Relief": "Happiness",
    "Shock": "Surprise",
    "Contempt": "Disgust",
    "Wonder": "Surprise",
    "Pride": "Happiness",
    "Proud": "Happiness",
    "Disdain": "Disgust",
    "Annoyance": "Anger",
    "Cynicism": "Disgust",
    "Cynical": "Disgust",
    "Irritation": "Anger",
    "Defiance": "Anger",
    "Wistfulness": "Sadness",
    "Concern": "Neutral",
    "Disappointment": "Surprise",
    "Resignation": "Neutral",
    "Skepticism": "Neutral",
    "Curiosity": "Neutral",
    "Urgency": "Neutral",
    "Urgent": "Neutral",
    "Suspense": "Neutral",
    "Suspicion": "Neutral",
    "Apology": "Neutral",
    "Nostalgia": "Neutral",
    "Determination": "Neutral",
    "Awe": "Neutral",
    "Incredulity": "Surprise",
    "Incredulous": "Surprise",
    "Serious": "Neutral",
    "Reflective": "Neutral",
    "Embarrassment": "Neutral",
    "Shyness": "Neutral",
    "Respectful": "Neutral",
    "Solemnity": "Neutral",
    "Tender": "Neutral",
    "Tenderness": "Neutral",
    "Empathetic": "Neutral",
    "Confident": "Neutral",
    "Confidence": "Neutral",
    "Assertive": "Neutral",
    "Defensiveness": "Neutral",
    "Empathy": "Neutral",
    "Hesitation": "Fear",
    "Exhaustion": "Neutral",
    "Warmth": "Happiness",
    "Hope": "Neutral",
    "Hopeful": "Neutral",
    "Regret": "Sadness",
}


def request_vllm_server(prompt, temperature=0.2, max_tokens=2048):
    # Extract VLLM server IP and PORT from environment variables
    vllm_server_ip = os.getenv("VLLM_SERVER_IP")
    vllm_server_port = os.getenv("VLLM_SERVER_PORT")
    url = f"http://{vllm_server_ip}:{vllm_server_port}/v1/chat/completions"

    try:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, json=payload, timeout=600, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()

        # Extract the model's response
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            return content
        else:
            print("No response from VLLM server.")
            return None

    except Exception as e:
        print(f"Error requesting VLLM server: {e}")
        return None


def is_valid_SAP(s: str):
    # Check if it is a string
    if not isinstance(s, str):
        return False, None
    # Check start and end, remove code block delimiters
    # Remove potential leading/trailing newlines with strip
    content = s.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].strip()
    if "```" in content:
        content = content.split("```", 1)[0].strip()

    # Define allowed categories
    AGE_CATEGORIES = {"Child", "Adult", "Elderly"}
    GENDER_CATEGORIES = {"Female", "Male"}
    EMOTION_CATEGORIES = {"Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"}

    try:
        data = json.loads(content)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return False, None

    # Top-level fields
    if not isinstance(data, dict):
        return False, None

    if set(data.keys()) != {"transcription", "paralinguistics", "nonLinguisticEvents"}:
        return False, None

    # transcription: str or None
    if data["transcription"] is not None and not isinstance(data["transcription"], str):
        return False, None

    # paralinguistics: dict
    paralinguistics = data["paralinguistics"]
    if not isinstance(paralinguistics, dict):
        return False, None
    if set(paralinguistics.keys()) != {"age", "gender", "emotion", "accent", "prosody", "timbre"}:
        return False, None

    # age
    age = paralinguistics["age"]
    if age is not None:
        if not isinstance(age, str):
            return False, None
        elif age not in AGE_CATEGORIES:
            print(f"Invalid age value: {age}")
            print(content)
            return False, None

    # gender
    gender = paralinguistics["gender"]
    if gender is not None:
        if not isinstance(gender, str):
            return False, None
        elif gender not in GENDER_CATEGORIES:
            print(f"Invalid gender value: {gender}")
            print(content)
            return False, None

    # emotion
    emotion = paralinguistics["emotion"]
    if emotion is not None:
        if not isinstance(emotion, str):
            return False, None
        elif emotion not in EMOTION_CATEGORIES:
            # Using mapped values
            if emotion in EMOTION_MAPPING:
                emotion = EMOTION_MAPPING[emotion]
            else:
                print(f"Invalid emotion value: {emotion}")
                print(content)
                print("Using 'Neutral' instead.")
                emotion = "Neutral"
            data["paralinguistics"]["emotion"] = emotion

    # accent: str or None
    accent = paralinguistics["accent"]
    if accent is not None and not isinstance(accent, str):
        return False, None

    # prosody: str or None
    prosody = paralinguistics["prosody"]
    if prosody is not None and not isinstance(prosody, str):
        return False, None

    # timbre: str or None
    timbre = paralinguistics["timbre"]
    if timbre is not None and not isinstance(timbre, str):
        return False, None

    # nonLinguisticEvents: dict
    if not isinstance(data["nonLinguisticEvents"], dict):
        return False, None
    if set(data["nonLinguisticEvents"].keys()) != {"description", "discreteEvents", "continuousEvents"}:
        return False, None

    # description: str or None
    description = data["nonLinguisticEvents"]["description"]
    if description is not None and not isinstance(description, str):
        return False, None

    # discreteEvents: list
    discreteEvents = data["nonLinguisticEvents"]["discreteEvents"]
    if not isinstance(discreteEvents, list):
        return False, None
    for event in discreteEvents:
        if not isinstance(event, dict):
            return False, None
        if set(event.keys()) != {"label", "characteristic"}:
            return False, None
        if not isinstance(event["label"], str):
            return False, None
        if not isinstance(event["characteristic"], str):
            return False, None

    # continuousEvents: list
    continuousEvents = data["nonLinguisticEvents"]["continuousEvents"]
    if not isinstance(continuousEvents, list):
        return False, None
    for event in continuousEvents:
        if not isinstance(event, dict):
            return False, None
        if set(event.keys()) != {"label", "characteristic"}:
            return False, None
        if not isinstance(event["label"], str):
            return False, None
        if not isinstance(event["characteristic"], str):
            return False, None

    return True, data


def load_prompt_template(prompt_file):
    """
    Load prompt template

    Args:
        prompt_file: Prompt file path

    Returns:
        Prompt template string
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def get_single_llm_judge_response(prompt_file, generated, reference, temperature=0.2, max_tokens=50):
    prompt_template = load_prompt_template(prompt_file)
    prompt = prompt_template.replace("${generated}", generated).replace("${reference}", reference)
    response = request_vllm_server(prompt, temperature=temperature, max_tokens=max_tokens)
    return response


def get_average_llm_judge_score(prompt_file, generated, reference, num_votes=3, temperature=0.2, max_tokens=50):
    scores = []
    for _ in range(num_votes):
        response = get_single_llm_judge_response(prompt_file, generated, reference, temperature, max_tokens)
        if response is not None:
            try:
                score = float(response.strip())
                scores.append(score)
            except ValueError:
                continue
    if len(scores) == 0:
        return (None, scores)
    return (sum(scores) / len(scores), scores)


def get_voted_llm_judge_response(prompt_file, generated, reference, num_votes=3, temperature=0.2, max_tokens=50):
    yes_count = 0
    total_count = 0
    for _ in range(num_votes):
        response = get_single_llm_judge_response(prompt_file, generated, reference, temperature, max_tokens)
        if response is not None:
            total_count += 1
            response_lower = response.strip().lower()
            if response_lower == "yes":
                yes_count += 1
    if yes_count > 0.5 * total_count:
        return ("yes", yes_count / total_count)
    elif yes_count < 0.5 * total_count:
        return ("no", yes_count / total_count)
    else:
        return (None, None if total_count == 0 else yes_count / total_count)


def caption2json(prompt_file, caption, temperature=0.0, max_tokens=2048):
    prompt_template = load_prompt_template(prompt_file)
    prompt = prompt_template.replace("${Input}", caption)
    response = request_vllm_server(prompt, temperature=temperature, max_tokens=max_tokens)
    return response
