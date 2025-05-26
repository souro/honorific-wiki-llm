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
sourabh.info()

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
df = sampled_df.copy()

# %%
language = "bengali"

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

        self.base_prompt_old = """বাঙলা ভাষায় উত্তর দিন।
        Write a detailed and vivid paragraph in {language} language about a completely fictional character with a made-up name, who {present_tense} {fame} {gender} figure {era} and {present_tense} a {role}. 
        The story should be entirely imaginary, with no references to real people, places, or historical events. This {gender_noun} {present_tense} {age} and {life_status}. {capital_subject} {existence} and {present_tense} {culture_origin} to the demography of the {language}-language speakers. 
        Describe {possessive} background, notable actions, and reputation, emphasizing {possessive} role in {timeline}. 
        Invent events that shaped {possessive} legacy and discuss {possessive} impact on society, ensuring all names and details are unique and fabricated. 
        
        **Important: You must write the paragraph entirely in the {language} language, without using a single English word. Do not provide any English translation.**
        Output format:
        <paragraph in pure {language} goes here — do not include a translation>
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
            fame=format_fame(attributes['Fame']),
            era=format_era(attributes['Era']),
            gender=pronouns['gender_noun'],
            gender_noun=pronouns['gender_noun'],
            role=self.get_role(attributes['Role']),
            subject=pronouns['subject'],
            object=pronouns['object'],
            possessive=pronouns['possessive'],
            capital_subject=pronouns['subject'].capitalize(),
            age=attributes['Age Group'].lower(),
            life_status=self.format_life_status(attributes['Life Status'], verb_tense),
            # ethnicity=attributes['Ethnicity'],
            # educational_qual=self.format_educational_qual(attributes['Educational Qualification']),
            # caste=attributes['Caste'].lower(),
            # nationality=attributes['Nationality'],
            # religion=attributes['Religion'].lower(),
            existence=self.format_existence(attributes['Entity'], attributes['Existence'], verb_tense),
            culture_origin=self.format_culture_origin(attributes['Origin of Culture']),
            # pronoun_usage=attributes['Pronoun/Verb Usage in Written Context'].lower(),
            present_tense=verb_tense['present'],
            past_tense=verb_tense['past'],
            timeline=self.format_timeline(attributes['Era'], attributes['Life Status']),
            name=attributes['Name']
        )

# Example usage:
generator = PromptGenerator()
# Example usage:
prompt_list = []

for i, row in df.iterrows():
    # if row["Article_Text"] == "":
    #   print(f"Skipping row {i}: Empty prompt")
    #   prompt_list.append("")
    #   continue

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
        'Name': row['Article_Title']
    }
    prompt = generator.generate_prompt(attributes)
    prompt_list.append(prompt)
    # print(prompt)
    # print("------------")

    # if i % 10 == 9: break
df['Generated Prompt'] = prompt_list

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
from google import genai
from google.genai import types
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

    elif model_name in ["Gemini-2.0-Flash", "Gemini-1.5-Pro"]: # Adjusted names slightly for clarity
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY not set."
        gemini_model_id = "gemini-2.0-flash-exp" if "Flash" in model_name else "models/gemini-1.5-pro-latest"
        if model_name not in _gemini_client_cache:
            print(f"Initializing Gemini client for {model_name} for the first time...")
            try:
                _gemini_client_cache[model_name] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                print(f"Gemini client for {model_name} initialized.")
            except Exception as e:
                return f"Error initializing Gemini client {model_name}: {e}"
        # else:
            # print(f"Using cached Gemini client for {model_name}.")

        client_gemini = _gemini_client_cache[model_name]
        # print(f"Accessing {model_name} for prompt: {prompt[:50]}...")
        try:
            # Ensure genai.types.GenerateContentConfig is available or use direct parameters
            # For simplicity, using direct parameters if config object causes issues.
            # Most common parameters are directly available.
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                max_output_tokens=1000,  # Adjust as needed
                temperature=temperature,
            )
            
            # print("Prompt:", prompt)
            
            response = client_gemini.models.generate_content(model=gemini_model_id,
                contents=prompt,
                config=generate_content_config
            )
            
            # print("Response:", response.text)
            
            return response.text
        except Exception as e:
            # Check for blocked content, which is a common "successful" failure mode
            if hasattr(e, 'response') and e.response.prompt_feedback.block_reason:
                 return f"Error with {model_name}: Blocked - {e.response.prompt_feedback.block_reason.name}. {str(e)}"
            if hasattr(response, "text"): # if response was generated but error happened after
                return response.text
            elif hasattr(response, "parts"): # check parts if text not available
                all_parts_text = ""
                for part in response.parts:
                    if hasattr(part, "text"):
                        all_parts_text += part.text
                if all_parts_text:
                    return all_parts_text
                else:
                    return f"Error with {model_name}: No text in response parts. Error: {e}"
            return f"Error with {model_name}: {e}"

    # ==== Hugging Face Hosted Models (Local Inference) ====
    else:
        # Note: For Llama models, ensure you are logged in: `huggingface-cli login`
        # Some models require `trust_remote_code=True`
        hf_mapping = {
            # Model Name: (HuggingFace_Model_ID, requires_trust_remote_code)
            "Aya": ("CohereLabs/aya-101", False),
            "AryaBhatta": ("GenVRadmin/AryaBhatta-GemmaUltra-Merged", True), # Assuming it might, common for custom code
            "Krutrim": ("krutrim-ai-labs/Krutrim-2-instruct", True), # Assuming it might
            "BharatGPT": ("sumanthd17/BharatGPT-3B-v0.1", True), # Assuming it might
            "Qwen2.5": ("Qwen/Qwen2.5-7B-Instruct", True), # Qwen models typically need it
            "Llama-3.1-8B-Instruct": ("meta-llama/Meta-Llama-3.1-8B-Instruct", True),
            "Mistral-7B-Instruct": ("mistralai/Mistral-7B-Instruct-v0.3", False), # Usually false for Mistral official
            "Mixtral-8x7B-Instruct": ("mistralai/Mixtral-8x7B-v0.1", False), # Usually false
            "DeepSeek": ("deepseek-ai/DeepSeek-R1", False)
        }

        if model_name not in hf_mapping:
            return f"Error: Model '{model_name}' not recognized in hf_mapping for local Hugging Face models."

        model_id_hf, trust_remote_code_hf = hf_mapping[model_name]
        quantize_hf = True # Defaulting to quantization for local models

        cache_key_user = (model_name, quantize_hf, trust_remote_code_hf) # Include trust_remote_code in user cache key
        if cache_key_user not in _hf_user_friendly_cache:
            # print(f"Model {model_name} (HF ID: {model_id_hf}) not in user-friendly cache. Loading...")
            try:
                tokenizer, model_ = load_model_hf(model_id_hf, quantize=quantize_hf, trust_remote_code_hf=trust_remote_code_hf)
                _hf_user_friendly_cache[cache_key_user] = (tokenizer, model_)
            except Exception as e:
                return f"Error loading HF model {model_name} ({model_id_hf}): {e}"
        else:
            # print(f"Using cached model {model_name} (HF ID: {model_id_hf}) from user-friendly cache.")
            tokenizer, model_ = _hf_user_friendly_cache[cache_key_user]

        # print(f"Generating text for HF model {model_name}...")
        messages = [{"role": "user", "content": prompt}]

        try:
            # --- MODIFIED SECTION for apply_chat_template ---
            if model_name not in ["AryaBhatta", "Aya", "Krutrim"] : 
                tokenized_inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                )
            elif model_name == "Aya":
                tokenized_inputs = tokenizer.encode(
                    prompt,
                    return_tensors="pt",
                )
            elif model_name == "Krutrim": 
                messages = [{"role":'system','content':"You are an AI assistant."}, {"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                tokenized_inputs = tokenizer(prompt, return_tensors='pt')
                tokenized_inputs.pop("token_type_ids", None)                
            else: 
                input_prompt = """
                    ### Instruction:
                    Write a detailed and vivid paragraph in Hindi.

                    ### Input:
                    {}

                    ### Response:
                    {}""".format(prompt, "")
                tokenized_inputs = tokenizer(
                    input_prompt,
                    return_tensors="pt",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                )
            
            # print(tokenized_inputs)
            
            # print(f"Tokenized inputs for model {model_name}: {tokenized_inputs}")
            # response_text = tokenizer.decode(**tokenized_inputs)
            # print(f"{response_text}")
            
            # print(tokenized_inputs)
            input_ids = tokenized_inputs.to(device)
            # attention_mask = tokenized_inputs['attention_mask'].to(model_.device)
            # --- END MODIFIED SECTION ---

            # Debugging prints (optional, but helpful)
            # print(f"Prompt length (tokens): {input_ids.shape[-1]}")
            # if hasattr(model_.config, 'max_position_embeddings'):
            #     print(f"Model max position embeddings: {model_.config.max_position_embeddings}")
            # if hasattr(tokenizer, 'model_max_length'):
            #     print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")
            # print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")


            # Check if total length will exceed model's capacity
            # max_possible_len = input_ids.shape[-1] + max_new_tokens_hf
            # model_max_len = model_.config.max_position_embeddings if hasattr(model_.config, 'max_position_embeddings') else tokenizer.model_max_length

            # if max_possible_len > model_max_len:
            #     print(f"Warning: Prompt length ({input_ids.shape[-1]}) + max_new_tokens ({max_new_tokens_hf}) = {max_possible_len} exceeds model max length ({model_max_len}). Truncating max_new_tokens.")
            #     # Adjust max_new_tokens to fit, ensuring at least a few tokens can be generated
            #     adjusted_max_new_tokens = model_max_len - input_ids.shape[-1] - 5 # -5 for safety buffer
            #     if adjusted_max_new_tokens < 10 : # if not enough space to generate anything meaningful
            #         return f"Error: Prompt is too long ({input_ids.shape[-1]} tokens) for the model's max length ({model_max_len}) to generate a response."
            #     max_new_tokens_hf = adjusted_max_new_tokens
            #     print(f"Adjusted max_new_tokens_hf to: {max_new_tokens_hf}")

            model_.to(device)

            # with torch.no_grad():
            if model_name not in ["Krutrim"] :
                model_.eval()
                with torch.no_grad() : 
                    outputs = model_.generate(
                        input_ids,
                        # attention_mask=attention_mask,
                        max_new_tokens=1000,
                        do_sample=True,
                        temperature=temperature,
                    )
            else:
                outputs = model_.generate(
                        **input_ids,
                        # attention_mask=attention_mask,
                        max_new_tokens=1000,
                        temperature=temperature,
                    )
                
            # print(outputs)
            
            generated_ids = outputs
            response_text = tokenizer.batch_decode(generated_ids)
            
            # print(f"Generated text for model {model_name}.")
            return response_text
        except Exception as e:
            print(f"Error during text generation with HF model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating with {model_name}: {e}"


# %%
# # Ensure the target column exists
# if "Response" not in df.columns:
#     df["Response"] = ""
# from huggingface_hub import login
# login()
from vllm import LLM, SamplingParams
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
        prompt_chat = hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
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
model_n = model_key = "Gemma"  # Set your model name here
output_csv_file_path = f"{model_n}_wo_names.csv"

outcome_df = pd.read_csv(output_csv_file_path) if os.path.exists(output_csv_file_path) else df.copy()
outcome_df.info()

batch_size = 25
num_generations = 5
model_path = vllm_models[model_key] if model_key in vllm_models else model_key

if model_key not in vllm_models:
    for i in tqdm(range(0, len(outcome_df))):
        # print("Response_4" in outcome_df.columns)
        # print(pd.notnull(outcome_df.at[i, "Response_4"]))
        if "Response_4" in outcome_df.columns and pd.notnull(outcome_df.at[i, "Response_4"]):
            # print("Skipping row {i}: Already completed")
            continue
        
        # print(f"{outcome_df.at[i, "Response_4"]}")
        
        if outcome_df.at[i, "Generated Prompt"] == "":
            continue
        
        prompt = outcome_df.at[i, "Generated Prompt"]
        
        # Check if the prompt is empty
        if not prompt:
            print(f"Skipping row {i}: Empty prompt")
            continue
        
        for j in range(num_generations):
            response = inference(prompt, model_path, "na" , temperature=0.1, max_tokens=1000)
            outcome_df.at[i, f"Response_{j}"] = response
        request_count += 1
        outcome_df.to_csv(output_csv_file_path, index=False)
        
        
else: 
    for start_idx in tqdm(range(0, len(outcome_df), batch_size)):
        batch_df = outcome_df.iloc[start_idx:start_idx + batch_size]
        prompts = batch_df["Generated Prompt"].tolist()

        # Check for already completed rows
        skip_mask = []
        valid_prompts = []
        row_indices = []
        for idx, prompt in zip(batch_df.index, prompts):
            if "Response_4" in outcome_df.columns and pd.notnull(outcome_df.at[idx, "Response_4"]):
                continue
            if not prompt:
                continue
            valid_prompts.append(prompt)
            row_indices.append(idx)

        if not valid_prompts:
            continue

        for i in range(num_generations):
            responses = inference_vllm_batch(valid_prompts, model_path, temperature=0.1, max_tokens=1000)

            for idx, response in zip(row_indices, responses):
                outcome_df.at[idx, f"Response_{i}"] = response

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



