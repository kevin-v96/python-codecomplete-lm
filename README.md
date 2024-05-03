# python-codecomplete-lm
Some small LMs (less than or around 1B parameters) for next token prediction in Python

## Models
Models I tried (you can find instructions about each in their own folders):

### First model: DeepSeek Coder 1.3B Parameter-efficient fine-tune
The [evalplus](https://evalplus.github.io/leaderboard.html) leaderboard has a list of the best current LLMs for code. For our size requirements (close to ~1B), the [DeepSeek Coder 1.3B base model](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) was ideal. I followed the steps given [here](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) to fine-tune models in a Parameter-efficient way using methods like LoRA (Low Rank Adapters that are aligned with and trained instead of the model weights, which are frozen) and QLoRA (LoRA that works on Quantized LLMs, like those loaded in 4bits with bitsandbytes). This took about ~2 hours of training. 

### Second model: GPT2 from scratch
To start, I trained a small GPT2 model (~124M parameters) from scratch using the data we have. This Huggingface [tutorial](https://huggingface.co/learn/nlp-course/chapter7/6) goes into how to do this. This is cumbersome to do in some ways - training a tokenizer (even the fast tokenizers from HF written in Rust) takes a few hours since tokenizer training is CPU-bound. 

So I used the tokenizer from the above model. This took a few hours off our total training time. Then I followed the standard training procedure outlined in the tutorial with different sizes of data, trying different dataset sizes to see what's realistic to train on a free Google Clab and get decent results.

### Third model: Decoder-only Transformer from scratch in PyTorch
I tried a few models completely from scratch (in PyTorch and using some libraries like x-transformers) but have left them as a to-do for a lack of time. 

## Eval
Here are some examples of vibes-based evals from the two models:

## DeepSeek:
### Example 1
Prompt:
```
# Load a dataset using pandas
import pandas as pd
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
```

DeepSeek base model output:
```
# Load a dataset using pandas
import pandas as pd
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
data.head()

# Display the last few rows of the dataset
data.tail()

# Display the number of rows and columns in the dataset
data.shape

# Display the data types of each column
data.dtypes

# Display the summary statistics of the dataset
data.describe()

# Display the number of unique values in each column
data.nunique()

# Display the number of missing values in each column
data.isnull().sum()

# Display the number of unique values in each column
data.nunique()

# Display the number of missing values in each column
data.isnull().sum()

# Display the number of unique values in each column
data.nunique()

# Display the number of missing values in each column
data.isnull().sum()

# Display the number of unique values in each column
data.nunique()

# Display the number of missing values in each column
data.is
```

DeepSeek Finetuned Model output:
```
# Load a dataset using pandas
import pandas as pd
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
data.head(10)

# Get information about the dataset
data.info()

# Get the index
for i in data.index:
    print(i)

# Get the number of observations
n_obs = data.shape[0]
print(n_obs)

# Get the number of columns
n_cols = data.shape[1]
print(n_cols)

# Get the list of columns
cols = data.columns
print(cols)

# Select a single column
sel_col = data['Country']
print(sel_col)

# Select multiple columns
sel_cols = data[['Country','Age']]
print(sel_cols)

# Select rows
# You can select by index, or by passing a list of indices
# By index, you can select by country using the following example
country_df = data.loc[data['Country']=='
```

### Example 2

Prompt:
```
# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a machine learning model (e.g., Random Forest)

```
DeepSeek Base model output:

```
# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a machine learning model (e.g., Random Forest)

# Fit the model to the training data

# Make predictions on the test data

# Evaluate the model's performance

# Use the model to make predictions on new data
<jupyter_output>
<empty_output>

```

DeepSeek Finetuned model output:

```

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a machine learning model (e.g., Random Forest)

# Split data into features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize an empty list to store accuracy scores
accuracies = []

# Iterate over different values of the regularization parameter for penalization
for alpha in [0.01, 0.1, 1, 10, 100, 1000]:
    # Instantiate a logistic regression model
    logreg = LogisticRegression(C=alpha, random_state=42)
    
    # Fit logistic regression to the training data
    logreg.fit(X

```
### GPT2 examples
Prompt:
```
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
```

Output:
```
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
data = np.random.randn(100)
#
```
Prompt:
```
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
```


## Outputs
- https://huggingface.co/MadMarx37/deepseek-coder-1.3b-python-peft
- https://huggingface.co/MadMarx37/python-gpt2
- This repo
- WandB report on finetuning the DeepSeek model: https://api.wandb.ai/links/kevinv3796/nzt0ndnq 

## References
- https://evalplus.github.io/leaderboard.html
- https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base
- https://huggingface.co/learn/nlp-course/chapter7/6
- https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html
- https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb
- https://huggingface.co/docs/transformers/training
- https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
- https://wandb.ai/matt24/vit-snacks-sweeps/reports/Hyperparameter-Search-for-HuggingFace-Transformer-Models--VmlldzoyMTUxNTg0
- https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html 
- https://github.com/lucidrains/x-transformers 
- https://huggingface.co/docs/datasets/en/use_with_pytorch 
- https://ai.plainenglish.io/creating-and-exploring-gpt-from-scratch-ffe84ac415a9
 