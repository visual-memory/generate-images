from pathlib import Path
import sys

PROMPT_ENHANCER_INFERENCE_PATH = (
    Path(__file__).parent / "3rdparty" / "PromptEnhancer" / "inference"
)
sys.path.insert(0, str(PROMPT_ENHANCER_INFERENCE_PATH))

from prompt_enhancer import HunyuanPromptEnhancer

models_root_path = "/raid/aluno_paulosantana/models/promptenhancer-7b/reprompt"

enhancer = HunyuanPromptEnhancer(models_root_path=models_root_path, device_map="auto")

# Enhance a prompt (Chinese or English)
user_prompt = "Third-person view, a race car speeding on a city track..."
new_prompt = enhancer.predict(
    prompt_cot=user_prompt,
    # Default system prompt is tailored for image prompt rewriting; override if needed
    temperature=0.7,   # >0 enables sampling; 0 uses deterministic generation
    top_p=0.9,
    max_new_tokens=256,
)

print("Enhanced:", new_prompt)
