import torch
from transformers import pipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('prompt', metavar='prompt',
                     action='store', help='Prompt to complete', type=str)

model_name = "MadMarx37/python-gpt2"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model = model_name, device = device
)

def generate_output(prompt):
    return pipe(prompt, num_return_sequences=1)[0]["generated_text"]

if __name__ == "__main__":
    args = parser.parse_args()
    prompt = args.prompt
    print(generate_output(prompt))