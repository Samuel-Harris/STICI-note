from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer


def construct_hf_model(model_id: str, quantisation_config: BitsAndBytesConfig) -> (AutoTokenizer, AutoModelForCausalLM):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantisation_config)

    return tokenizer, model
