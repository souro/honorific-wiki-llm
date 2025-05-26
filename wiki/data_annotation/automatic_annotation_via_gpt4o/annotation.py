import os
import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def init_openai_client():
    api_key = os.getenv("AZURE_OPENAI_KEY", "")
    endpoint = os.getenv("ENDPOINT_URL", "")
    deployment = os.getenv("DEPLOYMENT_NAME", "")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-05-01-preview")
    return client, deployment

def generate_output(client, deployment, prompt):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500
    )
    return response

def strip_response(prompt_response):
    return prompt_response.choices[0].message.content.strip()

def get_pronoun_and_verb_examples(language):
    if language == 'Bengali':
        pronouns = '''- **Examples of Honorific Pronouns**: "তাঁরা", "তিনি", "তাঁর", "তাঁদের"\n- **Examples of Non-Honorific Pronouns**: "সে", "তারা", "ও", "ওদের"'''
        verbs = '''- **Examples of Honorific Verbs**: "করেছেন", "বলেছেন"\n- **Examples of Non-Honorific Verbs**: "করেছে", "বলেছে"'''
    elif language == 'Hindi':
        pronouns = '''- **Examples of Honorific Pronouns**: "वे", "उनका", "उन्होंने"\n- **Examples of Non-Honorific Pronouns**: "वह", "उसका", "उसने"'''
        verbs = '''- **Examples of Honorific Verbs**: "उन्होंने किया", "उन्होंने कहा"\n- **Examples of Non-Honorific Verbs**: "उसने किया", "उसने कहा"'''
    else:
        pronouns, verbs = None, None
    return pronouns, verbs

def create_prompt(language, title, text):
    pronouns, verbs = get_pronoun_and_verb_examples(language)
    return f'''
We need your help to specify the usage of honorific and non-honorific pronouns, and/or verbs in the {language} Wikipedia article titled "{title}".

{pronouns}
{verbs}

**Article Text:**

{text}

**Task:**

Please carefully review the article text and select the value for the following features for the article:

- **Entity:** God, Human, Animal, or Other Being
- **Fame (Sentiment associated with the entity):** Famous, Infamous, Controversial
- **Age Group:** Juvenile (under 18), Adult (18-60), Old (60+), or Not Applicable
- **Gender:** Male, Female, Gender Neutral/Non-Specific
- **Role:** Politics and Governance, Science and Technology, Education and Academia, Arts and Culture, Entertainment, Religion and Spirituality, Sports, Business and Economy, Military, Media and Journalism, Law and Justice, Medicine and Healthcare, Literature and Philosophy, Public Service, Activists and Reformers, Infamous/Controversial Activists, Royalty and Nobility, Deity, Criminals, Others
- **Origin of Culture:** Native or Exotic
- **Existence Type:** Real, Fictional, Mythological
- **Life Status:** Alive, Dead, or Not Applicable
- **Era:** Historical or Modern
- **Count:** Singular, Plural, or Not Applicable

**Pronoun/Verb in Wiki Article**: Honorific or Non-Honorific
**Pronoun/Verb in a Written Setting**: Honorific or Non-Honorific
**Pronoun/Verb in a Spoken Setting**: Honorific or Non-Honorific

**Output Format**:

ONLY provide your answers in dictionary format. Example:

{{
    'Entity': 'Human',
    'Fame': 'Famous',
    'Age Group': 'Adult',
    'Gender': 'Male',
    'Role': 'Politics and Governance',
    'Origin of Culture': 'Native',
    'Existence Type': 'Real',
    'Life Status': 'Dead',
    'Era': 'Modern',
    'Count': 'Singular',
    'Pronoun/Verb in Wiki Article': 'Honorific',
    'Pronoun/Verb in Written Setting': 'Honorific',
    'Pronoun/Verb in Spoken Setting': 'Non-Honorific'
}}
'''

def parse_response(response_text):
    explanation_index = response_text.find("### Explanation:")
    if explanation_index != -1:
        response_text = response_text[:explanation_index]
    response_text = response_text.strip('```python\n').strip('```').strip()
    response_text = response_text.strip('```json\n').strip('```').strip()
    response_text = response_text.replace("'", '"')
    return json.loads(response_text)

def annotate_articles(client, deployment, df, lang, output_file_path):
    results = []
    with open(output_file_path, "w", encoding="utf-8") as file:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            title = row.get('Title', 'Unknown Title')
            text = row.get('Introductory Text', '')

            if not text:
                logging.warning(f"[{idx}] Skipping article '{title}' due to empty text.")
                continue

            try:
                prompt = create_prompt(lang, title, text)
                if not prompt or not isinstance(prompt, str):
                    raise ValueError("Generated prompt is invalid.")

                prompt_response = generate_output(client, deployment, prompt)
                if not prompt_response or not isinstance(prompt_response, str):
                    raise ValueError("Prompt response is invalid.")

                file.write(f'article_title: {title}\n')
                file.write(f'prompt_response: {prompt_response}\n\n')

                raw_text = strip_response(prompt_response)
                if not raw_text or not isinstance(raw_text, str):
                    raise ValueError("Stripped response is invalid.")

                parsed = parse_response(raw_text)
                if not isinstance(parsed, dict):
                    raise ValueError("Parsed response is not a dictionary.")

                required_keys = [
                    'Entity', 'Fame', 'Age Group', 'Gender', 'Role',
                    'Origin of Culture', 'Existence Type', 'Life Status',
                    'Era', 'Count', 'Pronoun/Verb in Wiki Article',
                    'Pronoun/Verb in Written Setting', 'Pronoun/Verb in Spoken Setting'
                ]

                for key in required_keys:
                    parsed[key] = parsed.get(key, None)
                    if parsed[key] is not None:
                        parsed[key] = parsed[key].strip()

                result = {
                    'Language': lang,
                    'Article_Title': title,
                    'Article_Text': text,
                    **{key: parsed.get(key) for key in required_keys}
                }
                results.append(result)
                logging.info(f"[{idx}] Successfully processed article '{title}'.")

            except Exception as e:
                logging.error(f"[{idx}] Error processing article '{title}': {e}")

    return results


def save_results(results, output_csv):
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()

def main():
    setup_logger()
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    client, deployment = init_openai_client()
    results = annotate_articles(client, deployment, df, args.lang, args.output_txt)
    save_results(results, args.output_csv)

if __name__ == "__main__":
    main()