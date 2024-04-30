from transformers import (
    pipeline,
    logging,
)
import argparse

parser = argparse.ArgumentParser(description='Complete Python code')

parser.add_argument('prompt', metavar='prompt',
                     action='store', help='Prompt to complete', type=str)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

#replace the below with your own model in case you end up finetuning one yourself
model_name = "MadMarx37/deepseek-coder-1.3b-python-peft"

#model is cached when prediction is run, so you should be able to run it multiple times without a repeated download overhead
#you can clear it from the cache with:
#pip install -U "huggingface_hub[cli]"
#huggingface-cli delete-cache
def generate_output(input):
    # Run text generation pipeline with our next model
    pipe = pipeline(task="text-generation", model=model_name, max_length=256)
    result = pipe(input)
    return result[0]['generated_text']

if __name__ == "__main__":
    args = parser.parse_args()
    prompt = args.prompt

    output = generate_output(prompt)
    print(output)