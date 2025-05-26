import xml.etree.ElementTree as ET
import re
import mwparserfromhell
import pandas as pd
import urllib.parse
import os
import random
import argparse
import logging


def setup_logger():
    logging.basicConfig(
        format='%(asctime)s — %(levelname)s — %(message)s',
        level=logging.INFO
    )


def remove_templates(text):
    result = ''
    brace_level = 0
    i = 0
    while i < len(text):
        if text[i:i + 2] == '{{':
            brace_level += 1
            i += 2
        elif text[i:i + 2] == '}}' and brace_level > 0:
            brace_level -= 1
            i += 2
        elif brace_level == 0:
            result += text[i]
            i += 1
        else:
            i += 1
    return result


def clean_text(text, lang):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = remove_templates(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)

    image_prefixes = {
        'en': ['File:'],
        'hi': ['File:', 'चित्र:'],
        'bn': ['File:', 'চিত্র:']
    }
    category_prefixes = {
        'en': ['Category:'],
        'hi': ['Category:', 'श्रेणी:'],
        'bn': ['Category:', 'বিষয়শ্রেণী:']
    }

    img_prefixes = image_prefixes.get(lang, ['File:'])
    img_prefixes_escaped = [re.escape(prefix) for prefix in img_prefixes]
    text = re.sub(r'\[\[(' + '|'.join(img_prefixes_escaped) + ')[^\[\]]*\]\]', '', text, flags=re.IGNORECASE)

    cat_prefixes = category_prefixes.get(lang, ['Category:'])
    cat_prefixes_escaped = [re.escape(prefix) for prefix in cat_prefixes]
    text = re.sub(r'\[\[(' + '|'.join(cat_prefixes_escaped) + ')[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\[http[^\]]*\]', '', text)
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)
    text = text.replace('[[', '').replace(']]', '')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_introductory_text(text):
    text = re.sub(r'^((\{\{[^\{\}]*\}\}|\s|<!--.*?-->)+)', '', text, flags=re.MULTILINE | re.DOTALL)
    section_match = re.search(r'\n==[^=]', text)
    if section_match:
        return text[:section_match.start()].strip()
    return text.strip()


def get_categories(text, lang):
    category_prefixes = {
        'en': ['Category:'],
        'hi': ['Category:', 'श्रेणी:'],
        'bn': ['Category:', 'বিষয়শ্রেণী:']
    }
    cat_prefixes = category_prefixes.get(lang, ['Category:'])
    cat_prefixes_escaped = [re.escape(prefix) for prefix in cat_prefixes]
    pattern = r'\[\[(' + '|'.join(cat_prefixes_escaped) + r')(.+?)(?:\|.*?)?\]\]'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return [match[1].strip() for match in matches]


def find_text_element(elem):
    paths = [
        './revision/text',
        'revision/text',
        './text',
        'text',
        '{*}revision/{*}text',
        '{*}text',
        './/text'
    ]
    for path in paths:
        text_elem = elem.find(path)
        if text_elem is not None and text_elem.text:
            return text_elem
    for child in elem:
        if child.tag.endswith('text') and child.text:
            return child
        text_elem = find_text_element(child)
        if text_elem is not None:
            return text_elem
    return None


def extract_infobox(text, lang):
    wikicode = mwparserfromhell.parse(text)
    templates = wikicode.filter_templates()
    infobox_data = {}

    infobox_template_names = {
        'en': ['infobox'],
        'hi': ['infobox', 'जानकारी'],
        'bn': ['infobox', 'তথ্যছক']
    }

    lang_infobox_names = infobox_template_names.get(lang, ['infobox'])
    for template in templates:
        template_name = str(template.name).strip().lower()
        if any(name in template_name for name in lang_infobox_names):
            for param in template.params:
                key = str(param.name).strip()
                value = param.value.strip_code().strip()
                value = clean_text(value, lang)
                infobox_data[key] = value
            break
    return infobox_data


def serialize_infobox_data(infobox_data):
    sanitized_infobox = {}
    for key, value in infobox_data.items():
        sanitized_key = key.replace(';', ',')
        sanitized_value = value.replace(';', ',')
        sanitized_infobox[sanitized_key] = sanitized_value
    infobox_items = [f"{k}:{v}" for k, v in sanitized_infobox.items()]
    return ';'.join(infobox_items)


def is_valid_article(title, text, categories, infobox_data, lang):
    if ':' in title:
        return False
    return True


def get_pageviews(title, lang):
    encoded_title = urllib.parse.quote(title.replace(' ', '_'))
    pageview_url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{encoded_title}/daily/20230101/20230131"
    return pageview_url


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_file', type=str, required=True, help='Path to input XML dump')
    parser.add_argument('--lang', type=str, required=True, help='Language code (e.g., en, hi, bn)')
    parser.add_argument('--output_csv', type=str, required=True, help='Output path for the CSV file')
    return parser.parse_args()


def process_articles(xml_file, lang, output_csv):
    logging.info(f"Parsing XML file: {xml_file}")
    context = ET.iterparse(xml_file, events=("start", "end"))
    context = iter(context)

    _, root = next(context)
    titles, intros, cats, infos = [], [], [], []

    for event, elem in context:
        if event == "end" and elem.tag.endswith("page"):
            title_elem = elem.find('title')
            title = title_elem.text if title_elem is not None else ''
            text_elem = find_text_element(elem)
            text = text_elem.text if text_elem is not None else ''
            categories = get_categories(text, lang)
            infobox_data = extract_infobox(text, lang)
            if is_valid_article(title, text, categories, infobox_data, lang):
                clean = clean_text(text, lang)
                intro = get_introductory_text(clean)
                serialized_infobox = serialize_infobox_data(infobox_data)
                titles.append(title)
                intros.append(intro)
                cats.append(','.join(categories))
                infos.append(serialized_infobox)
            root.clear()

    logging.info(f"Total valid articles extracted: {len(titles)}")
    df = pd.DataFrame({'title': titles, 'intro': intros, 'categories': cats, 'infobox': infos})
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved cleaned data to {output_csv}")


def main():
    setup_logger()
    args = parse_args()
    logging.info("Starting Wikipedia XML processing")
    process_articles(args.xml_file, args.lang, args.output_csv)
    logging.info("Processing complete")


if __name__ == "__main__":
    main()