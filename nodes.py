from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click


class LoadHiggsAudioModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "bosonai/higgs-audio-v2-generation-3B-base"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL_PATH",)
    FUNCTION = "load_model"
    CATEGORY = "Higgs Audio"

    def load_model(self, model_path):
        MODEL_PATH = model_path
        
        return (MODEL_PATH,)


class LoadHiggsAudioTokenizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "bosonai/higgs-audio-v2-tokenizer"}),
            }
        }

    RETURN_TYPES = ("AUDIOTOKENIZER",)
    RETURN_NAMES = ("AUDIO_TOKENIZER_PATH",)
    FUNCTION = "load_model"
    CATEGORY = "Higgs Audio"

    def load_model(self, model_path):
        AUDIO_TOKENIZER_PATH = model_path
        
        return (AUDIO_TOKENIZER_PATH,)




system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
