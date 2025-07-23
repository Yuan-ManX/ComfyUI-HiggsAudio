from .nodes import LoadHiggsAudioModel, LoadHiggsAudioTokenizer, LoadHiggsAudioSystemPrompt, LoadHiggsAudioPrompt, HiggsAudio, SaveHiggsAudio

NODE_CLASS_MAPPINGS = {
    "LoadHiggsAudioModel": LoadHiggsAudioModel,
    "LoadHiggsAudioTokenizer": LoadHiggsAudioTokenizer,
    "LoadHiggsAudioSystemPrompt": LoadHiggsAudioSystemPrompt,
    "LoadHiggsAudioPrompt": LoadHiggsAudioPrompt,
    "HiggsAudio": HiggsAudio,
    "SaveHiggsAudio": SaveHiggsAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHiggsAudioModel": "Load Higgs Audio Model",
    "LoadHiggsAudioTokenizer": "Load Higgs Audio Tokenizer",
    "LoadHiggsAudioSystemPrompt": "Load Higgs Audio System Prompt",
    "LoadHiggsAudioPrompt": "Load Higgs Audio Prompt",
    "HiggsAudio": "Higgs Audio",
    "SaveHiggsAudio": "Save Higgs Audio",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
