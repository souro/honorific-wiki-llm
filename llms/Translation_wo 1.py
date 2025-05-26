
# %%
# Install necessary packages
# !pip install pandas numpy openai google-generativeai transformers torch accelerate 

# %% [markdown]
# Data
from pdb import set_trace as bp
# %%
import time
import pandas as pd
# import google.generativeai as genai
# from google.colab import drive
import os

# %%
# Install the required libraries
# !pip install gspread google-auth

# Import libraries
# import gspread
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# from google.auth import default
import pandas as pd

columns_rqd = ["Article_Text","Article_Title","Entity","Fame","Age Group","Gender","Role","Origin of Culture","Existence Type","Life Status","Era","Count","Pronoun/Verb in Wiki Article","Pronoun/Verb in Spoken Setting","Pronoun/Verb in Written Setting"]

# Authenticate and create a client
# auth.authenticate_user()
# creds, _ = default()
# gc = gspread.authorize(creds)

def convert_df(df):
  # data_list = worksheet.get_all_values()
  # print(data_list[0])
  # if not data_list:
  #   df = pd.DataFrame()
    # print("Worksheet appears empty.")
  # else:
    # df = pd.DataFrame(data_list[1:], columns=data_list[0])
    # print(df.columns)

  if "introductory_text" in df.columns :
    df["Article_Text"] = df["introductory_text"]

  df = df[columns_rqd]
  # df = df.set_index(df["Article Name (English)"])
  # df = df.reset_index(drop=True)
  # print(df.index)
  # print(df.index.is_unique)
  # print(df)
  return df

# Open the Google Sheets file (replace 'your_spreadsheet_name' with your file's name)
sourabh = pd.read_csv("wiki_honorifics_human_annotation_Saurabh.csv")
# print(sourabh)
# sourabh = pd.DataFrame(convert_df(sourabh))
sourabh = convert_df(sourabh)
# sourabh.info()

#wiki_honorifics_human_annotation_Atharva
atharva = pd.read_csv("wiki_honorifics_human_annotation_Atharva.csv")
# atharva = pd.DataFrame(atharva.get_all_records())
atharva = convert_df(atharva)
# atharva.info()

#wiki_honorifics_human_annotation_Arif
arif = pd.read_csv("wiki_honorifics_human_annotation_Arif.csv")
arif = convert_df(arif)
# arif.info()

#wiki_honorifics_human_annotation_Shivam
shivam = pd.read_csv("wiki_honorifics_human_annotation_Shivam.csv")
shivam = convert_df(shivam)
# shivam.info()

#wiki_honorifics_human_annotation_Mukund
mukund = pd.read_csv("wiki_honorifics_human_annotation_Mukund.csv")
mukund = convert_df(mukund)
# mukund.info()

#wiki_honorifics_human_annotation_Soumya
soumya = pd.read_csv("wiki_honorifics_human_annotation_Soumya.csv")
soumya = convert_df(soumya)
# soumya.info()

#wiki_honorifics_human_annotation_Soumya
souro = pd.read_csv("wiki_honorifics_human_annotation_Souro.csv")
souro = convert_df(souro)

sougata = pd.read_csv("wiki_honorifics_human_annotation_Sougata.csv")
sougata = convert_df(sougata)

sampled_df = pd.read_csv("sampled_data_bn.csv")

# hi_wiki = gc.open("wiki_10k_annotations_hi_bn").get_worksheet(1)
# print(hi_wiki)
# hi_wiki = convert_df(hi_wiki)

# be_wiki = gc.open("wiki_10k_annotations_hi_bn").get_worksheet(2)
# print(be_wiki)
# be_wiki = convert_df(be_wiki)


# %%
hi_human1 = pd.concat([sourabh, soumya, arif], ignore_index=True)
hi_human2 = pd.concat([sourabh, atharva, shivam], ignore_index=True)

be_human1 = souro
be_human2 = sougata

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# df = sampled_df.copy()
import torch

# %%
model_n = model_key = "GPT-4o"  # Set your model name here
language = "bengali"
translate_file = f"Results/{model_n}_english_{language}_wo_names.csv"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU 1
device = "cuda:0" if torch.cuda.is_available() else "cpu"


translate_df = pd.read_csv(translate_file)
df = translate_df.copy()

# %% [markdown]
# Prompt Generation

# %%
def format_era(era):
    """Format the era into a natural-sounding phrase."""
    if era == "Modern":
        return "from the modern era(post 1800)"
    elif era == "Historical":
        return "from the historical era(pre 1800)"
    return era.lower()

def format_fame(fame):
    """Format the fame level into a natural-sounding phrase."""
    if fame == "Famous":
        return "a famous"
    elif fame == "In Famous":
        return "an infamous"
    elif fame == "Controversial":
        return "a controversial"
    else: 
        return ""

def get_verb_tense(life_status, entity_type, existence):
    """
    Determine the appropriate verb tense based on life status and entity type.

    Args:
        life_status (str): "Alive", "Dead", or "Not Applicable"
        entity_type (str): The type of entity (e.g., "Human & Real", "God")

    Returns:
        dict: Dictionary containing present and past tense forms of 'be'
    """
    if life_status == "Dead":
        return {"present": "was", "past": "was", "exist": "existed"}
    elif life_status == "Alive":
        return {"present": "is", "past": "was", "exist": "exists"}
    elif "God" in entity_type or life_status == "Not Applicable":
        return {"present": "is", "past": "has been", "exist": "exists"}
    return {"present": "is", "past": "was"}

class PromptGenerator:
    def __init__(self):
        self.pronoun_mappings = {
            'Boy/Man': {
                'subject': 'he',
                'object': 'him',
                'possessive': 'his',
                'possessive_pronoun': 'his',
                'reflexive': 'himself',
                'gender_noun': 'male'
            },
            'Girl/Woman': {
                'subject': 'she',
                'object': 'her',
                'possessive': 'her',
                'possessive_pronoun': 'hers',
                'reflexive': 'herself',
                'gender_noun': 'female'
            },
            'Non-Binary': {
                'subject': 'they',
                'object': 'them',
                'possessive': 'their',
                'possessive_pronoun': 'theirs',
                'reflexive': 'themselves',
                'gender_noun': 'person'
            },
            'Gender Neutral': {
                'subject': 'they',
                'object': 'them',
                'possessive': 'their',
                'possessive_pronoun': 'theirs',
                'reflexive': 'themselves',
                'gender_noun': 'person'
            },
            'Transgender': {
                'subject': 'they',
                'object': 'them',
                'possessive': 'their',
                'possessive_pronoun': 'theirs',
                'reflexive': 'themselves',
                'gender_noun': 'person'
            },
            'Not Applicable': {
                'subject': 'it',
                'object': 'it',
                'possessive': 'its',
                'possessive_pronoun': 'its',
                'reflexive': 'itself',
                'gender_noun': 'entity'
            }
        }

        self.base_prompt = """Write a detailed and vivid paragraph in {language} about a {fame} {gender} figure {era} who {present_tense} a member of {role}.
This {gender_noun} {present_tense} {age} and {life_status}, {present_tense} identified as {ethnicity} by ethnicity, and {past_tense} {educational_qual}.
{subject} {present_tense} from a {caste} background in {nationality}, where {subject} {past_tense} practiced {religion}.
{subject} {past_tense} existed in {existence} and {present_tense} {culture_origin} to the {language}-speaking culture.
Describe {possessive} background, notable actions, and reputation, emphasizing {possessive} role in {timeline}.
Highlight the events that shaped {possessive} legacy and discuss {possessive} impact on society."""

        self.base_prompt_old = """
        The following is a detailed and vivid paragraph in {language} about the entity in the paragraph. Translate the paragraph into {language}.
        Paragraph: {LLM_Response}
        """

    def get_pronouns(self, gender):
        return self.pronoun_mappings.get(gender, self.pronoun_mappings['Gender Neutral'])

    def format_existence(self, entity, existence_type, existence_tense):


        # existence_mappings = {
        #     'Human & Real': 'reality',
        #     'Human & Fictional character': 'fiction',
        #     'Human & Mythological character': 'mythology',
        #     'Animal & Real': 'the animal kingdom',
        #     'Animal & Fictional character': 'fictional fauna',
        #     'Animal & Mythological character': 'mythological beasts',
        #     'God': 'the divine realm',
        #     'Other Being': 'supernatural existence'
        # }

        exist = existence_tense["exist"]
        verb = existence_tense["present"]


        ## Use switch case or if else since entity and existence are now separate
        if existence_type == "Real" and entity == "Human":
            return f"{exist} in reality"
        elif existence_type == "Fictional" and entity == "Human":
            return f"{verb} a fictional character"
        elif existence_type == "Mythological" and entity == "Human":
            return f"{verb} a fictional character"
        elif existence_type == "Real" and entity == "Animal":
            return f"{verb} a member of the animal kingdom"
        elif existence_type == "Fictional" and entity == "Animal":
            return f"{verb} a fictional fauna"
        elif existence_type == "Mythological" and entity == "Animal":
            return f"{verb} a mythological beast"
        elif existence_type == "God":
            return f"{exist} in the divine realm"
        elif existence_type == "Other Being":
            return f"{verb} a supernatural being"

        return "reality"



    def format_life_status(self, status, verb_tense):
        if status == "Alive":
            return f"{verb_tense['present']} living"
        else:
            return "no longer living"
        return "eternal"

    def format_educational_qual(self, qual):
        if qual == "Not Applicable":
            return "not formally educated"
        return f"educated to the {qual.lower()} level"

    def format_culture_origin(self, origin):
        if origin == "Native":
            return "native"
        return "foreign"

    def format_timeline(self, era, life_status):
        if life_status == "Dead" or era == "Historical":
            return "history"
        return "contemporary times"

    def get_role(self, role):
        if role == "Deity":
          return "deity"
        elif role == "Royalty and Nobility":
          return "member of the royalty and nobility class"
        elif role == "Religion & Spirituality" or role == "Activists & Reformers":
          return f"member of the {role.lower()} community"
        elif role == "Politics and Governance":
          return f"is a part of the {role.lower()} landscape"
        elif role == "Sports":
          return "sportsperson"
        else:
          return "member of the " + role.lower() + " industry"

    def generate_prompt(self, attributes):
        """Generate a grammatically correct prompt based on provided attributes."""
        pronouns = self.get_pronouns(attributes['Gender'])
        verb_tense = get_verb_tense(attributes['Life Status'], attributes['Entity'], attributes['Existence'])

        return self.base_prompt_old.format(
            language=attributes['language'],
            # fame=format_fame(attributes['Fame']),
            # era=format_era(attributes['Era']),
            # gender=pronouns['gender_noun'],
            # gender_noun=pronouns['gender_noun'],
            # role=self.get_role(attributes['Role']),
            # subject=pronouns['subject'],
            # object=pronouns['object'],
            # possessive=pronouns['possessive'],
            # capital_subject=pronouns['subject'].capitalize(),
            # age=attributes['Age Group'].lower(),
            # life_status=self.format_life_status(attributes['Life Status'], verb_tense),
            # # ethnicity=attributes['Ethnicity'],
            # # educational_qual=self.format_educational_qual(attributes['Educational Qualification']),
            # # caste=attributes['Caste'].lower(),
            # # nationality=attributes['Nationality'],
            # # religion=attributes['Religion'].lower(),
            # existence=self.format_existence(attributes['Entity'], attributes['Existence'], verb_tense),
            # culture_origin=self.format_culture_origin(attributes['Origin of Culture']),
            # # pronoun_usage=attributes['Pronoun/Verb Usage in Written Context'].lower(),
            # present_tense=verb_tense['present'],
            # past_tense=verb_tense['past'],
            # timeline=self.format_timeline(attributes['Era'], attributes['Life Status']),
            name=attributes['Name'],
            LLM_Response=attributes['LLM_Response'],
        )

# Example usage:
generator = PromptGenerator()
# Example usage:
prompt_list_0 = []  
prompt_list_1 = []
prompt_list_2 = []
prompt_list_3 = []
prompt_list_4 = []

for i, row in df.iterrows():
    # if row["Article_Text"] == "":
    #   print(f"Skipping row {i}: Empty prompt")
    #   prompt_list.append("")
    #   continue
    for j in range(5): 
        attributes = {
            'language': language,
            'Fame': row['Fame'],
            'Gender': row['Gender'],
            'Era': row['Era'],
            'Role': row['Role'],
            'Age Group': row['Age Group'],
            'Life Status': row['Life Status'],
            # 'Ethnicity': row['Ethnicity'],
            # 'Educational Qualification': row['Educational Qualification'],
            # 'Caste': row['Caste'],
            # 'Nationality': row['Nationality'],
            # 'Religion': row['Religion'],
            'Entity': row['Entity'],
            'Existence': row['Existence Type'],
            'Origin of Culture': row['Origin of Culture'],
            'Pronoun/Verb in Wiki Article': row['Pronoun/Verb in Wiki Article'],
            'Article_Text': row['Article_Text'],
            'Name': row['Article_Title'],
            'LLM_Response': row[f'Response_{j}'],
        }
        prompt = generator.generate_prompt(attributes)
        
        if j == 0:
            prompt_list_0.append(prompt)
        elif j == 1:
            prompt_list_1.append(prompt)
        elif j == 2:
            prompt_list_2.append(prompt)
        elif j == 3:
            prompt_list_3.append(prompt)
        elif j == 4:
            prompt_list_4.append(prompt)
    # print(prompt)
    # print("------------")

    # if i % 10 == 9: break
df[f'Prompt_0'] = prompt_list_0
df[f'Prompt_1'] = prompt_list_1
df[f'Prompt_2'] = prompt_list_2
df[f'Prompt_3'] = prompt_list_3
df[f'Prompt_4'] = prompt_list_4

# %%
df.info()

# %%
df['Generated Prompt'].value_counts()

# %%
# !pip install 'git+https://github.com/huggingface/transformers.git' bitsandbytes accelerate
# !pip install bitsandbytes
# # !huggingface-cli login
# !echo "export HF_HOME='/home/atharva.mehta/.cache/huggingface/hub'" >> ~/.bashrc
os.environ['HF_HUB_CACHE'] = '/home/atharva.mehta/.cache/huggingface/hub'
os.environ['HF_HOME'] = '/home/atharva.mehta/.cache/huggingface/hub'
# !echo $HF_HUB_CACHE
# !pip install -U bitsandbytes

# %%
# from google import genai
# from google.genai import types

os.environ["GEMINI_API_KEY"] = ""
# Retrieve the API key
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key not found! Please set GEMINI_API_KEY as an environment variable.")

# Configure Gemini API
# genai.configure(api_key=api_key)

# %%
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, AutoModelForSeq2SeqLM
import openai
# from google import genai
# from google.genai import types
# import accelerate # Good to have, imported by transformers if needed for device_map="auto"
import time
import pandas as pd # Assuming df is a pandas DataFrame for the driver code

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU 1
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# --- API Key Setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("Warning: OPENAI_API_KEY not set. OpenAI models will not be available.")

if GEMINI_API_KEY:
    # genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API key set.")
else:
    print("Warning: GEMINI_API_KEY not set. Gemini models will not be available.")

# --- Global Model and Client Caches ---
_hf_model_id_cache = {} # For (tokenizer, model) tuples, keyed by (model_id_hf, quantize)
_hf_user_friendly_cache = {} # For (tokenizer, model) tuples, keyed by (user_friendly_name, quantize)
_gemini_client_cache = {} # For Gemini GenerativeModel instances, keyed by user-friendly name

# --- Hugging Face Model Loading ---
def load_model_hf(model_id_hf, quantize=False, trust_remote_code_hf=False):
    """
    Loads a Hugging Face model and tokenizer.
    Caches based on the Hugging Face model ID and quantization state.
    """
    try: 
        cache_key = (model_id_hf, quantize)
        if cache_key not in _hf_model_id_cache:
            print(f"Loading HF model {model_id_hf} with quantize={quantize}, trust_remote_code={trust_remote_code_hf} for the first time...")
            
            quantization_config = None
            if quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16 if preferred and supported
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            tokenizer = AutoTokenizer.from_pretrained(model_id_hf)
            
            # Ensure pad token is set for open-ended generation, common fix
            # if tokenizer.pad_token is None:
            #     if tokenizer.eos_token is not None:
            #         print(f"Tokenizer for {model_id_hf} has no pad_token. Setting pad_token to eos_token.")
            #         tokenizer.pad_token = tokenizer.eos_token
            #     else:
            #         print(f"Warning: Tokenizer for {model_id_hf} has no pad_token and no eos_token. Adding a new pad token.")
            #         tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            if model_id_hf not in ["CohereLabs/aya-101"]:
                print(f"Loading model {model_id_hf} as a Causal LM model.")
                model = AutoModelForCausalLM.from_pretrained(model_id_hf)
            else: 
                print(f"Loading model {model_id_hf} as a Seq2Seq model.")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id_hf)
            
            _hf_model_id_cache[cache_key] = (tokenizer, model)
            print(f"Finished loading HF model {model_id_hf}.")
        else:
            print(f"Using cached HF model {model_id_hf} (quantize={quantize}, trust_remote_code={trust_remote_code_hf}).")
    except Exception as e:
        print(f"Error loading HF model {model_id_hf}: {e}")
        return None, None
    
    return _hf_model_id_cache[cache_key]

# --- Main Inference Function ---
def inference(prompt: str, model_name: str, api_base:str, max_tokens=512, temperature=0.1) -> str:
    """
    Generates text using the specified model.

    Args:
        prompt (str): The input prompt.
        model_name (str): The user-friendly name of the model.
        max_new_tokens_hf (int): Max new tokens for Hugging Face models.
        temperature_hf (float): Temperature for Hugging Face models.

    Returns:
        str: The generated text or an error message.
    """
    # ==== API-Based Models ====
    if model_name == "GPT-4o":
        # if not OPENAI_API_KEY:
        #     return "Error: OPENAI_API_KEY not set."
        print(f"Accessing GPT-4o for prompt: {prompt[:50]}...")
        try:
            # Updated to use the new client syntax
            
            client = openai.AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-05-01-preview",
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000, # Max output tokens for OpenAI
                temperature=temperature,
            )
            
            usage = response.usage  # {'prompt_tokens': ..., 'completion_tokens': ..., 'total_tokens': ...}

            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # GPT-4o pricing (as of May 2024 OpenAI pricing)
            # Input: $0.005 per 1K tokens, Output: $0.015 per 1K tokens
            input_cost_per_token = 2.5 / 1000000
            output_cost_per_token = 10 / 1000000

            total_cost = (prompt_tokens * input_cost_per_token) + (completion_tokens * output_cost_per_token)

            print(f"Tokens used - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            print(f"Estimated cost: ${total_cost:.6f}")
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with GPT-4o: {e}"
        

# %%
# # Ensure the target column exists
# if "Response" not in df.columns:
#     df["Response"] = ""
# from huggingface_hub import login
# login()
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

vllm_cache = {}

def inference_vllm_batch(prompt_list, model_name, temperature=0.1, max_tokens=1000):
    """
    Batched inference for a list of prompts using local vLLM LLM object.
    Returns a list of generated outputs.
    """

    if model_name not in vllm_cache:
        model = LLM(model=model_name, dtype="auto", max_model_len=4096)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        vllm_cache[model_name] = (model, hf_tokenizer)

    model, hf_tokenizer = vllm_cache[model_name]

    # Format prompts with chat template
    prompts = []
    for prompt in prompt_list:
        if model_name not in ["Sarvam", "AryaBhatta", "Aya", "Krutrim"] :
            prompt_chat = hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

        else:
            prompt_chat = hf_tokenizer.tokenize(
                prompt,
                return_tensors="pt",
            )
        
        prompts.append(prompt_chat)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,  # one output per prompt
        stop=["</s>"]
    )

    outputs = model.generate(prompts, sampling_params)

    # Sort by request index to preserve order
    # outputs = sorted(outputs, key=lambda o: o.request_id)
    return [o.outputs[0].text for o in outputs]



# New: Unified inference wrapper for vLLM models
import requests

def inference_vllm(prompt, model_name, api_base="http://localhost:8000/v1", temperature=0.1, max_tokens=1000):
    """
    Uses OpenAI-compatible vLLM API server to perform inference.
    Assumes the model_name has been loaded into the vLLM API server.
    """
    model, tokenizer, hf_tokenizer = load_model_and_tokenizer(model_name)
    prompt_order = [{"role": "user", "content": prompt}]
    prompt = hf_tokenizer.apply_chat_template(prompt_order, tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens = max_tokens,
        n=1
    )
    
    outputs = model.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text 
 

# Plug into your existing loop:
vllm_models = {
    "Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "Aya": "CohereLabs/aya-101",
    "Krutrim": "krutrim-ai-labs/Krutrim-2-instruct",
    "BharatGPT": "CoRover/BharatGPT-3B-Indic",
    "AryaBhatta": "GenVRadmin/AryaBhatta-GemmaUltra-Merged",
    "Sarvam": "sarvamai/sarvam-1",
    "Gemma": "google/gemma-7b-it"
    # Add any additional vLLM-loaded names
}


# %%
# Function to generate output using Gemini 1.5


def generate_output(prompt):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model = "models/gemini-2.0-flash",
            contents = prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=5000,
                temperature=0.7
            )
        )

        print(response)
        return response.text if hasattr(response, "text") else "Error: No response text"
    except Exception as e:
        print(str(e))
        return f"Error: {str(e)}"

# Process rows while respecting rate limits
request_count = 0
MAX_REQUESTS_PER_DAY = 1050
INTERVAL = 1  # Wait time between requests (in seconds)

from tqdm import tqdm
import time
import pandas as pd
# from IPython.display import display
# import ipywidgets as widgets
import warnings

# Suppress specific user warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

# Progress bar setup
# progress = widgets.IntProgress(value=0, min=0, max=len(df), description='Progress:', bar_style='info')
# label = widgets.Label(value="Starting...")
# box = widgets.VBox([progress, label])
# display(box)

request_count = 0  # Initialize this somewhere if not already
output_csv_file_path = f"Translated_{model_n}_wo_names.csv"

outcome_df = pd.read_csv(output_csv_file_path) if os.path.exists(output_csv_file_path) else df.copy()
outcome_df.info()

batch_size = 50
num_generations = 5
model_path = vllm_models[model_key] if model_key in vllm_models else model_key

if model_key not in vllm_models:
    for i in tqdm(range(0, len(outcome_df))):
        # print("Response_4" in outcome_df.columns)
        # print(pd.notnull(outcome_df.at[i, "Response_4"]))
        if "Translate_4" in outcome_df.columns and pd.notnull(outcome_df.at[i, "Translate_4"]):
            # print("Skipping row {i}: Already completed")
            continue
        
        # print(f"{outcome_df.at[i, "Response_4"]}")
        
        
        
        for j in range(num_generations):
            if outcome_df.at[i, f"Prompt_{j}"] == "":
                continue
            
            prompt = outcome_df.at[i, f"Prompt_{j}"]
            
            # Check if the prompt is empty
            if not prompt:
                print(f"Skipping row {i}: Empty prompt")
                continue
            response = inference(prompt, model_path, "na" , temperature=0.1, max_tokens=1000)
            outcome_df.at[i, f"Translate_{j}"] = response
        request_count += 1
        outcome_df.to_csv(output_csv_file_path, index=False)
        
        
else: 
    for start_idx in tqdm(range(0, len(outcome_df), batch_size)):
        batch_df = outcome_df.iloc[start_idx:start_idx + batch_size]
        

        for i in range(num_generations):
            prompts = batch_df[f"Prompt_{i}"].tolist()

            # Check for already completed rows
            skip_mask = []
            valid_prompts = []
            row_indices = []
            for idx, prompt in zip(batch_df.index, prompts):
                if "Translate_4" in outcome_df.columns and pd.notnull(outcome_df.at[idx, "Translate_4"]):
                    continue
                if not prompt:
                    continue
                valid_prompts.append(prompt)
                row_indices.append(idx)

            if not valid_prompts:
                continue
            responses = inference_vllm_batch(valid_prompts, model_path, temperature=0.1, max_tokens=1000)

            for idx, response in zip(row_indices, responses):
                outcome_df.at[idx, f"Translate_{i}"] = response

        request_count += len(valid_prompts)
        outcome_df.to_csv(output_csv_file_path, index=False)

        if request_count >= MAX_REQUESTS_PER_DAY:
            break


# label.value = "Done!"
# progress.bar_style = 'success'


# %%
# df["Response"].value_counts()

# %%
# Save the updated DataFrame back to CSV
outcome_df.to_csv(output_csv_file_path, index=False)
print(f"Updated CSV saved to {output_csv_file_path}")

# %%



