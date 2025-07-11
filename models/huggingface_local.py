from transformers import pipeline

def run_huggingface_model_locally(text: str, model_name: str = "distilgpt2"):
    """
    Generates text using a Hugging Face model locally.
    Requires a GPU for larger models for reasonable speed.
    """
    try:
        # Load a text generation pipeline
        # You can specify a different model_name here (e.g., "stabilityai/stablelm-zephyr-3b")
        # Be aware of memory requirements for larger models.
        generator = pipeline("text-generation", model=model_name)
        
        results = generator(text, max_length=100, num_return_sequences=1)
        return results[0]['generated_text']
    except Exception as e:
        print(f"Error running Hugging Face model locally: {e}")
        print("Consider a smaller model or ensuring you have sufficient hardware.")
        return None

if __name__ == "__main__":
    prompt = "The quick brown fox jumps over the lazy dog. A new adventure begins when"
    generated_text = run_huggingface_model_locally(prompt)
    if generated_text:
        print("\nGenerated Text (Hugging Face Local):")
        print(generated_text)
