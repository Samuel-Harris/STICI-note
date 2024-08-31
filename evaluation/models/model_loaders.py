from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler


def load_llama_cpp_model(model_path: str, temperature: float, top_p: float) -> LlamaCpp:
    callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])

    return LlamaCpp(model_path=model_path,
                    temperature=temperature,
                    max_tokens=2000,
                    top_p=top_p,
                    callback_manager=callback_manager,
                    verbose=True,
                    n_ctx=2048,
                    n_batch=512,
                    n_gpu_layers=-1,
                    f16_kv=True,
                    )
