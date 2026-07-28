git submodule update --init --recursive
hf download tencent/HunyuanImage-2.1 --include "reprompt/*" --local-dir /raid/aluno_paulosantana/models/promptenhancer-7b
hf download PromptEnhancer/PromptEnhancer-32B --local-dir /raid/aluno_paulosantana/models/promptenhancer-32b
uv sync --extra prompt-enhancing
source .venv/bin/activate

PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0,1 --push-pr

enhance prompt
screen -L -Logfile screen.log -dmS enhance_shards bash -lc 'PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 2,2,2,2,3,3,3,3 --push-pr'

gen images
screen -L -Logfile gen.log -dmS gen_shards bash -lc 'PYTHONPATH=src uv run python -m generate_images.generate_persona_images_shards run --gpus 2,3 --push-pr'
screen -L -Logfile gen_2.log -dmS gen_shards_2 bash -lc 'PYTHONPATH=src uv run python -m generate_images.generate_persona_images_shards run --gpus 0 --push-pr'
