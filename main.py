from pathlib import Path
import sys

PROMPT_ENHANCER_INFERENCE_PATH = (
    Path(__file__).parent / "3rdparty" / "PromptEnhancer" / "inference"
)
sys.path.insert(0, str(PROMPT_ENHANCER_INFERENCE_PATH))

from prompt_enhancer import HunyuanPromptEnhancer
from prompt_enhancer_v2 import PromptEnhancerV2

# models_root_path = "/raid/aluno_paulosantana/models/promptenhancer-7b/reprompt"
# enhancer = HunyuanPromptEnhancer(models_root_path=models_root_path, device_map="auto")

models_root_path = "/raid/aluno_paulosantana/models/promptenhancer-32b"
enhancer = PromptEnhancerV2(models_root_path=models_root_path, device_map="auto")

def enhance_prompt(user_prompt):
    print("Original:", user_prompt)
    new_prompt = enhancer.predict(
        prompt_cot=user_prompt,
        # Default system prompt is tailored for image prompt rewriting; override if needed
        # temperature=0.7,   # >0 enables sampling; 0 uses deterministic generation
        # top_p=0.9,
        # max_new_tokens=256,
    )
    print("Enhanced:", new_prompt)

user_prompt_1 = """Generate a photo of a person with the following self-description:
I am also a musician on the weekends.
I love playing video games.
Love to read drama books.
Hey there my name is jordan and i am a veterinarian.
I am originally from california but i live in florida."""

user_prompt_1_1 = """Generate a photo of a person with the following characteristics:
I am also a musician on the weekends.
I love playing video games.
Love to read drama books.
Hey there my name is jordan and i am a veterinarian.
I am originally from california but i live in florida."""

user_prompt_1_2 = """I am also a musician on the weekends.
I love playing video games.
Love to read drama books.
Hey there my name is jordan and i am a veterinarian.
I am originally from california but i live in florida."""

# user_prompt_2 = """Generate a photo of a person with the following self-description:
# I work in a factory.
# I am not social.
# I do not eat well.
# I sleep most of the day."""

# user_prompt_3 = """Generate a photo of a person with the following self-description:
# I like to race rc cars.
# I like to play nintendo.
# I live in the great white north.
# I have a pet husky."""

enhance_prompt(user_prompt_1)
enhance_prompt(user_prompt_1_1)
enhance_prompt(user_prompt_1_2)
# enhance_prompt(user_prompt_2)
# enhance_prompt(user_prompt_3)