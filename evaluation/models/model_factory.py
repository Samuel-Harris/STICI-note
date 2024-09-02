from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.language_models import LLM


def construct_hf_model(model_path: str) -> LLM:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    return LlamaCpp(model_path=model_path,
                    temperature=0.75,
                    max_tokens=2000,
                    top_p=1,
                    callback_manager=callback_manager,
                    verbose=True,
                    n_ctx=2048,
                    n_batch=512,
                    n_gpu_layers=-1,
                    f16_kv=True,
                    )
