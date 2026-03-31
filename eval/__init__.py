
# Llama family
LLAMA_3_2_3B_INSTRUCT = "meta-llama/llama-3.2-3b-instruct"
LLAMA_3_1_8B_INSTRUCT = "meta-llama/llama-3.1-8b-instruct"
LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"
# LLAMA_3_1_405B_INSTRUCT = "meta-llama/llama-3.1-405b-instruct"  # EXPENSIVE! $5 WON'T LAST FOR LONG

LLAMA_MODELS = [
    LLAMA_3_2_3B_INSTRUCT,
    LLAMA_3_1_8B_INSTRUCT,
    LLAMA_3_3_70B_INSTRUCT,
    # LLAMA_3_1_405B_INSTRUCT,
]

# GPT family
GPT_5_MINI = "openai/gpt-5-mini"
GPT_4_1_MINI = "openai/gpt-4.1-mini"

GPT_MODELS = [
    GPT_5_MINI,
    GPT_4_1_MINI,
]

# Qwen family
QWEN3_235B_A22B_2507 = "qwen/qwen3-235b-a22b-2507"
QWEN3_30B_A3B_INSTRUCT_2507 = "qwen/qwen3-30b-a3b-instruct-2507"
QWEN3_CODER_30B_A3B_INSTRUCT = "qwen/qwen3-coder-30b-a3b-instruct"

QWEN_MODELS = [
    QWEN3_235B_A22B_2507,
    QWEN3_30B_A3B_INSTRUCT_2507,
    QWEN3_CODER_30B_A3B_INSTRUCT,
]

ALL_MODEL_NAMES = LLAMA_MODELS + GPT_MODELS + QWEN_MODELS

def get_all_model_names_filtered(
    include_llama405b : bool=False,
    include_llama_family : bool=True,
    include_gpt_family : bool=True,
    include_qwen_family : bool=True
    ):
    if include_llama405b:
        raise Exception("For now, to save money, we won't use Llama405B!")

    returned_names = []
    if include_llama_family:
        returned_names.extend(LLAMA_MODELS)
    if include_gpt_family:
        returned_names.extend(GPT_MODELS)
    if include_qwen_family:
        returned_names.extend(QWEN_MODELS)
    return returned_names