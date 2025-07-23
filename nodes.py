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


class LoadHiggsAudioSystemPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("SYSTEMPROMPT",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Higgs Audio"

    def load_prompt(self, text):
        system_prompt = text
        
        return (system_prompt,)


class LoadHiggsAudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Higgs Audio"

    def load_prompt(self, text):
        prompt = text
        
        return (prompt,)


class HiggsAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL_PATH": ("MODEL",),
                "AUDIO_TOKENIZER_PATH": ("AUDIOTOKENIZER",),
                "system_prompt": ("SYSTEMPROMPT",),
                "prompt": ("PROMPT",),
                "max_new_tokens": ("INT", {"default": 1024}),
                "temperature": ("FLOAT", {"default": 0.3}),
                "top_p": ("FLOAT", {"default": 0.95}),
                "top_k": ("INT", {"default": 50}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "generate"
    CATEGORY = "Higgs Audio"

    def generate(self, MODEL_PATH, AUDIO_TOKENIZER_PATH, system_prompt, prompt, max_new_tokens, temperature, top_p, top_k, device):
        
        messages = [
            Message(
                role="system",
                content=system_prompt,
            ),
            Message(
                role="user",
                content=prompt,
            ),
        ]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
        
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        
        return (output,)


