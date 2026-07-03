git submodule update --init --recursive
hf download tencent/HunyuanImage-2.1 --include "reprompt/*" --local-dir /raid/aluno_paulosantana/models/promptenhancer-7b
uv sync --extra prompt-enhancing
CUDA_VISIBLE_DEVICES=1 python main.py