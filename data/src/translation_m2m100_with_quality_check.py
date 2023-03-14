"""
Copyright 2023, LAION contributors, inclduing Ontocord, LLC
and the other authors of OIG
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import json

# translation based stuff
try:
    import transformers
except:
    os.system("pip install transformers  sentencepiece")

from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    BertModel,
    BertTokenizerFast,
)
import langid


import torch
from torch.nn.functional import cosine_similarity
import string

punc = string.punctuation + "¿？,،、º。゜ "

# We assume we are running on CPU only
# use labse to do a comparison
try:
    if m2m100_model is None:
        pass
except:
    # m2m100_model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").half().eval().to('cuda')
    # m2m100_tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
    m2m100_model = (
        M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        .half()
        .eval()
        .to("cuda")
    )
    m2m100_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    labse_model = (
        BertModel.from_pretrained("sentence-transformers/LaBSE")
        .half()
        .eval()
        .to("cuda")
    )
    labse_tokenizer = BertTokenizerFast.from_pretrained("sentence-transformers/LaBSE")

from collections import Counter


def get_ngram(sent, window_size=3, lang="en"):
    if lang in {"zh", "ja", "ko", "th"}:
        tokens = sent
        ret = [
            "".join(tokens[i : i + window_size])
            for i in range(len(tokens) - window_size)
        ]
    else:
        tokens = sent.split()
    ret = [
        " ".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
    ]
    return Counter(ret)


def high_ngram(sent, cutoff=0.09):
    aHash = get_ngram(sent)
    sent_len = sent.count(" ") + 1
    for key in list(aHash.keys()):
        aHash[key] = aHash[key] / sent_len
    return any(a > cutoff for a in aHash.values())


xglm_langs = {
    "en",
    "ru",
    "zh",
    "de",
    "es",
    "fr",
    "ja",
    "it",
    "pt",
    "el",
    "ko",
    "fi",
    "id",
    "tr",
    "ar",
    "vi",
    "th",
    "bg",
    "ca",
    "hi",
    "et",
    "bn",
    "ta",
    "ur",
    "sw",
    "te",
    "eu",
    "my",
    "ht",
    "qu",
}

langs2 = []
# TODO - add some multitrans, backtrans to get more diversity
# TODO - add option to return as paragraph by lang.
def get_translation_set(
    text,
    threshold=0.75,
    langs=[
        "af",
        "am",
        "ar",
        "ast",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "br",
        "bs",
        "ca",
        "ceb",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "ff",
        "fi",
        "fr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "ig",
        "ilo",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "lb",
        "lg",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "my",
        "ne",
        "nl",
        "no",
        "ns",
        "oc",
        "or",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sd",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "ss",
        "su",
        "sv",
        "sw",
        "ta",
        "th",
        "tl",
        "tn",
        "tr",
        "uk",
        "ur",
        "uz",
        "vi",
        "wo",
        "xh",
        "yi",
        "yo",
        "zh",
        "zu",
    ],
    return_original=False,
):
    ret = []
    if type(text) is str:
        text = [text]
    else:
        text = list(text)
    with torch.no_grad():
        labse_text = labse_tokenizer(text, padding=True, return_tensors="pt").to("cuda")
        en_embed = labse_model(**labse_text).pooler_output

        for target_lang in langs:

            input = m2m100_tokenizer(text, padding=True, return_tensors="pt").to("cuda")
            generated_tokens = m2m100_model.generate(
                **input, forced_bos_token_id=m2m100_tokenizer.get_lang_id(target_lang)
            )
            trans_text = m2m100_tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            labse_text = labse_tokenizer(
                trans_text, padding=True, return_tensors="pt"
            ).to("cuda")

            all_trans_embed = labse_model(**labse_text).pooler_output
            similarity = cosine_similarity(en_embed, all_trans_embed, dim=1)
            # trs = []
            for sim, tr in zip(similarity, trans_text):
                # print (sim, tr)
                if sim >= threshold and not high_ngram(tr):
                    ret.append(tr)
                    # trs.append(tr)
            # if add_backtrans:
        if return_original:
            return set(text + ret)  # [a.lower() for a in ret])
        else:
            return set(ret)


xlgm_in_m2m = [
    "ar",
    "bg",
    "bn",
    "ca",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "ht",
    "id",
    "it",
    "ja",
    "ko",
    "my",
    "pt",
    "ru",
    "sw",
    "ta",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

multilingual_data = []
with open("/content/oa_v3_fixed_plus_safety.jsonl") as input:
    batch = []
    for l in input:
        data = json.loads(l.strip())
        multilingual_data.append({"text": data["text"], "lang": "en"})
        if "Python" in data["text"]:
            continue  # we need to make sure not to translate the functions and code themselves. only the comments
        if len(data["text"]) > 500:
            for lang in xlgm_in_m2m:
                if lang == "en":
                    continue
                trans = get_translation_set(data["text"], langs=[lang])
                for text2 in trans:
                    if langid.classify(text2)[0] != lang:
                        print("bad trans", text2)
                        continue
                    multilingual_data.append({"text": text2, "lang": lang})
        else:
            batch.append(data)
        if len(batch) >= 40:
            for lang in xlgm_in_m2m:
                if lang == "en":
                    continue
                trans = get_translation_set([d["text"] for d in batch], langs=[lang])
                for text2 in trans:
                    if langid.classify(text2)[0] != lang:
                        print("bad trans", text2)
                        continue
                    multilingual_data.append({"text": text2, "lang": lang})
            batch = []

if batch:
    for lang in xlgm_in_m2m:
        if lang == "en":
            continue
        trans = get_translation_set([d["text"] for d in batch], langs=[lang])
        for text2 in trans:
            if langid.classify(text2)[0] != lang:
                print("bad trans", text2)
                continue
            multilingual_data.append({"text": text2, "lang": lang})
    batch = []
