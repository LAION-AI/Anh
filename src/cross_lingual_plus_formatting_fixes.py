

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

import os
#you will need to download all the {lang}_oa_v3_fixed_plus_safety.jsonl files
if not os.path.exists("/content/es_oa_v3_fixed_plus_safety.jsonl"):
  os.system("/content/drive/Shareddrives/LAION/oa_translated/* /content")
x_lang_mapper = {
'zh': {'en': "用英语回答。", 'de':"用德语回答。",'fr': "用法语回答。", 'es':"用西班牙语回答。", 'hi':"用印地语回答。",'id':"印尼语回答。",'ja':"日语回答。", 'ms':"马来西亚语回答。", 'pt': "葡萄牙语回答。", 'ru':"俄语回答。",'vi':"越南语回答。", 'zh': "中文回答。"},
'de': {'en': "Antwort auf Englisch.", 'de': "Antwort auf Deutsch.", 'fr': "Antwort auf Französisch.", 'es': "Antwort auf Spanisch.", 'hi': "Antwort auf Hindi.", 'id': "Antwort auf Indonesisch", 'ja':"Antwort auf Japanisch", 'ms': "Antwort auf Malaysisch", 'pt': "Antwort auf Portugiesisch", 'ru': "Antwort auf Russisch", 'vi': " Antwort auf Vietnamesisch", 'zh': "Antwort auf Chinesisch"},
'es': {'en': "Respuesta en inglés.", 'de': "Respuesta en alemán.", 'fr': "Respuesta en francés.", 'es': "Respuesta en español.", 'hi': "Respuesta en hindi.", 'id': "Respuesta en indonesio", 'ja':"Respuesta en japonés", 'ms': "Respuesta en malasio", 'pt': "Respuesta en portugués", 'ru': "Respuesta en ruso", 'vi': " Respuesta en vietnamita", 'zh': "Respuesta en chino"},
'fr': {'en': "Répondez en anglais.", 'de':"Répondez en allemand.", 'fr': "Répondez en français.", 'es':"Répondez en espagnol.", 'hi':"Répondez en hindi.", 'id':"Répondez en indonésien", 'ja':"Répondre en japonais", 'ms':"Répondre en malaisien", 'pt':"Répondre en portugais", 'ru':"Répondre en russe", 'vi':"Répondre en vietnamien", 'zh': "Répondre en chinois"},
'hi': {'en': "अंग्रेज़ी में उत्तर दें।", 'de': "जर्मन में उत्तर दें।", 'fr': "फ़्रेंच में उत्तर दें।", 'es': "स्पेनिश में उत्तर दें।", 'हाय': " हिंदी में उत्तर दें।", 'id': "इंडोनेशियाई में उत्तर", 'ja': "जापानी में उत्तर", 'ms': "मलेशियाई में उत्तर", 'pt': "पुर्तगाली में उत्तर", 'ru': "रूसी में उत्तर", 'vi': "वियतनामी में उत्तर", 'zh': "चीनी में उत्तर"},
'id': {'en': "Jawab dalam bahasa Inggris.", 'de': "Jawab dalam bahasa Jerman.", 'fr': "Jawab dalam bahasa Prancis.", 'es': "Jawab dalam bahasa Spanyol.", 'hi': " Jawab dalam bahasa Hindi.", 'id': "Jawab dalam bahasa Indonesia", 'ja':"Jawab dalam bahasa Jepang", 'ms': "Jawab dalam bahasa Malaysia", 'pt': "Jawab dalam bahasa Portugis", 'ru': "Jawab dalam bahasa Rusia", 'vi': "Jawab dalam bahasa Vietnam", 'zh': "Jawab dalam bahasa China"},
'ms': {'en': "Jawapan dalam bahasa Inggeris.", 'de': "Jawapan dalam bahasa Jerman.", 'fr': "Jawapan dalam bahasa Perancis.", 'es': "Jawapan dalam bahasa Sepanyol.", 'hi': " Jawab dalam bahasa Hindi.", 'id': "Jawapan dalam bahasa Indonesia", 'ja':"Jawapan dalam bahasa Jepun", 'ms': "Jawapan dalam bahasa Malaysia", 'pt': "Jawapan dalam bahasa Portugis", 'ru': "Jawapan dalam bahasa Rusia", 'vi': "Jawapan dalam bahasa Vietnam", 'zh': "Jawapan dalam bahasa Cina"},
'pt': {'en': "Resposta em inglês.", 'de': "Resposta em alemão.", 'fr': "Resposta em francês.", 'es': "Resposta em espanhol.", 'hi': " Resposta em hindi.", 'id': "Resposta em indonésio", 'ja': "Resposta em japonês", 'ms': "Resposta em malaio", 'pt': "Resposta em português", 'ru': "Resposta em russo", 'vi': "Resposta em vietnamita", 'zh': "Resposta em chinês"},
'ja': {'en': "英語で答えてください。", 'de': "ドイツ語で答えてください。", 'fr': "フランス語で答えてください.", 'es': "スペイン語で答えてください.", 'hi': "ヒンディー語で回答", 'id': "インドネシア語で回答", 'ja':"日本語で回答", 'ms': "マレーシア語で回答", 'pt': "ポルトガル語で回答", 'ru': "ロシア語で回答", 'vi': "ベトナム語で回答", 'zh': "中国語で回答"},
'ru': {'en': "Ответить на английском", "de": "Ответить на немецком", "fr": "Ответить на французском", "es": "Ответить на испанском", "hi": " Ответить на хинди.", "id": "Ответить на индонезийском", "ja": "Ответить на японском", "ms": "Ответить на малазийском", "pt": "Ответить на португальском", "ru": "Ответить по-русски", "vi": "Ответить по-вьетнамски", "zh": "Ответить по-китайски"},
'vi': {'en': "Trả lời bằng tiếng Anh.", 'de': "Trả lời bằng tiếng Đức.", 'fr': "Trả lời bằng tiếng Pháp.", 'es': "Trả lời bằng tiếng Tây Ban Nha.", 'hi': " Trả lời bằng tiếng Hindi.", 'id': "Trả lời bằng tiếng Indonesia", 'ja':"Trả lời bằng tiếng Nhật", 'ms': "Trả lời bằng tiếng Malaysia", 'pt': "Trả lời bằng tiếng Bồ Đào Nha", 'ru': "Trả lời bằng tiếng Nga", 'vi': "Trả lời bằng tiếng Việt", 'zh': "Trả lời bằng tiếng Trung"}, 
'en': {'en': "Answer in English.", 'de': "Answer in German.", 'fr': "Answer in French.", 'es': "Answer in Spanish.", 'hi': "Answer in Hindi.", 'id': "Answer in Indonesian", 'ja':"Answer in Japanese", 'ms': "Answer in Malaysian", 'pt': "Answer in Portuguese", 'ru': "Answer in Russian", 'vi': "Answer in Vietnamese", 'zh': "Answer in Chinese"},
} 

import json, glob


def add_formating(orig_text, text, lang):
  if "def " in orig_text and "#" in orig_text:
    answer= orig_text.split("Assistant:",1)[-1]
    orig_answer = answer
    answer2 = text.split("Assistant:",1)[-1]
    answer2 = answer2.replace('“', '"').replace("”", '"').replace('""', '"""').replace('""""', '"""').replace('""""', '"""')
    if '#' in answer:
      #print (answer.split('#'))
      #print (answer2.split('#'))
      for s1, s2 in zip(answer.split('#'), answer2.split('#'), ):
        s1 = s1.strip('\n "').split("\n")[0]
        len_en = len(s1.split())
        if'(' in s1 and '[' in s1 or "<" in s1: len_en = len_en*2
        if lang in {'ja', 'zh'}:
          s2 = s2.strip('\n "')
          s2 = s2[:min(len(s2),len_en)]
        else:
          s2 = s2.split()
          s2 = " ".join(s2[:min(len(s2),len_en)])

        if '"""' in s2:
          s2 = s2.split('"""')[0]    
        if 'def' in s2 and 'def' not in s1:
          s2 = s2.split('def')[0]   
        if 'class' in s2 and 'class' not in s2:
          s2 = s2.split('class')[0]  
        if 'print' in s2 and 'print' not in s2:
          s2 = s2.split('print')[0]                                       
        #print (s1, ' ----- ', s2)
        answer = answer.replace(s1, s2)

    if '"""' in answer:
     
      i=0
      for s1, s2 in zip(answer.split('"""'), answer2.split('"""'), ):
        s1 = s1.strip('\n "')
        s2 = s2.strip('\n "')
        
        if i%2 == 0: 
          i+= 1
          continue
        i+=1
        answer = answer.replace(s1, s2)
    
    if "class " in answer and "class " in answer2:
      prefix, answer = answer.split("class ",1)
      prefix2, _ = answer2.split("class ",1)
      answer2 = prefix.strip('"') + "\n\n"+"class " + answer2
    if "def " in answer and "def " in answer2:
      prefix, answer = answer.split("def ",1)
      prefix2, _ = answer2.split("def ",1)
      answer = prefix2.strip('"') + "\n\n"+"def " + answer
    
    if orig_answer != answer:
      #print ('##')
      #print (answer)
      return answer
  if "\n1." in orig_text: 
    text = text.replace(" 1.", "\n1.").replace(" 2.", "\n2.").replace(" 3.", "\n3.").replace(" 4.", "\n4.").replace(" 5.", "\n5.").\
          replace(" 6.", "\n6.").replace(" 7.", "\n7.").replace(" 8.", "\n8. ").replace(" 9.", "\n9.").replace(" 10.", "\n10.").replace(" 11.", "\n11.").\
          replace(" 1 ", "\n1. ").replace(" 2 ", "\n2.").replace(" 3 ", "\n3. ").replace(" 4 ", "\n4. ").replace(" 5 ", "\n5. ").\
          replace(" 6 ", "\n6. ").replace(" 7 ", "\n7.").replace(" 8 ", "\n8. ").replace(" 9 ", "\n9. ").replace(" 10 ", "\n10. ").replace(" 11 ", "\n11. ")
  if "\n- "in orig_text:
    text = text.replace("- ", "\n- ")
  if "\n* "in orig_text:
    text = text.replace("* ", "\n* ")
  if "\n# "in orig_text:
    text = text.replace("# ", "\n# ")    
  return text
try:
  if aHash: pass
except:
  aHash = {}

import random

def create_cross_lingual(output, recreate_formatting=True):
  global aHash
  if not aHash or recreate_formatting:
    for file in glob.glob("*oa*.jsonl"):
      lang = file.split("/")[-1].split("_")[0]
      with open(file) as input:
        for l in input:
          data = json.loads(l.strip())
          #print (data)
          text = (data['text'])
          aHash[text] = bHash = aHash.get(text, {})
          translation = ""
          for tran in data['translate']:
            translation += "\nUser: " +  tran['human']+ "\nAssistant:" + tran['answer']
          translation = add_formating(text, translation, lang) 
          bHash[lang] = translation.strip()
  for english in aHash:
    if "world" not in english and ("US" in english or "U.S." in english or "United States" in english or "American" in english or "New York" in english or "city" in english or "federal" in english):
      continue
    translations = aHash[english]
    translations['en'] = english
    all_langs = list(translations.keys())
    for lang, trans in translations.items():
      x_lang = random.choice(all_langs)
      #print ("##")
      if x_lang == 'en' and lang == 'en': continue
      output.write (json.dumps({'text': trans, 'metadata': {'src_lang': lang, 'target_lang': lang, 'source': 'cross_lingual_plus_safety'}})+"\n")
      if x_lang == lang:
        continue
      else: 
        answer = translations[x_lang].split("Assistant:",1)[-1]
        #print (lang, x_lang)
        dialog = trans.split("\nAssistant:")[0]+" " + x_lang_mapper[lang][x_lang]
        if dialog[-1] not in {"。","."}:
          dialog = dialog + "."
        dialog += "\nAssistant: " + answer
        all_langs.remove(x_lang)
        #print (dialog)
        output.write (json.dumps({'text': dialog, 'metadata': {'src_lang': lang, 'target_lang': x_lang, 'source': 'cross_lingual_plus_safety'}})+"\n")

with open("cross_lingual.jsonl", "w") as output:
  create_cross_lingual(output)      
os.system("cp cross_lingual.jsonl /content/drive/Shareddrives/LAION/multilingual-bot")  
