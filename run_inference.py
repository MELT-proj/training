import json
import torch
from types import SimpleNamespace
import soundfile as sf
from melt.modeling import MELTForSequenceClassification, MELTProcessor
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer
from datasets import Audio, load_dataset

dataset = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev", split="dev")
dataset = dataset.cast_column("audio", Audio(decode=False))
BASEDIR = "/mnt/scratch-artemis/sonal/IWSLT26/data/audios/dev/"

# Prepare the output list
results = []


def processor(feature_extractor, tokenizer):
    """Create a MELTProcessor instance with proper MELTConfig."""

    return MELTProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


model_name = "/mnt/scratch-artemis/giuseppe/melt-data/outputs/MELT_QE_v1.0/" 
model = MELTForSequenceClassification.from_pretrained(model_name, text_decoder_kwargs={"num_labels": 1})
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", extra_special_tokens={"audio_token":"<|audio|>","audio_bos_token":"<|audio_bos|>","audio_eos_token":"<|audio_eos|>",})
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
processor = processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
)
print("Model Loaded")

# Ensure model is not None before proceeding
if model is None:
    raise ValueError("Model loading failed, please check the model path.")

for instance in dataset:
    doc_id = instance['doc_id']
    true = instance['score']
    # src_audio = instance['audio']['array']
    audio_path = BASEDIR + instance['doc_id'] + ".wav"
    src_audio_array, sample_rate = sf.read(audio_path, dtype='float32')
    if src_audio_array.ndim > 1:
        src_audio_array = src_audio_array.mean(axis=1)  # mono

    if sample_rate != 16000:
        waveform = torch.tensor(src_audio_array).unsqueeze(0)
        #waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        src_audio_array = waveform.squeeze().numpy()

    tgt_text = instance['tgt_text']
    tgt_lang = "de"
    
    text = f"{processor.audio_token} Score how well this {tgt_lang} translation matches the audio: {tgt_text}. Return a float between 0 and 1."
    inputs = processor(text=text, audio=src_audio_array, return_tensors="pt", padding=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.squeeze().item()

    print({"doc_id":doc_id, "tgt_text": tgt_text, "score": score/2.5, "true": true/100})
    results.append({"doc_id":doc_id, "tgt_text": tgt_text, "score": score/2.5, "true": true/100})

# # Write results to a JSONL file
output_file = "/mnt/scratch-artemis/sonal/IWSLT26/eval/inference_results.jsonl"
with open(output_file, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
