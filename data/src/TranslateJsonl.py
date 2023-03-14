# Translation script originally for Swedish, but can be used for any language supported by the HuggingFace translation pipeline
import jsonlines
import torch
from transformers import pipeline
from termcolor import colored

model_name = "Helsinki-NLP/opus-mt-en-sv"
tokenizer = pipeline(
    "translation", model=model_name, device=0 if torch.cuda.is_available() else -1
).tokenizer
translator = pipeline(
    "translation", model=model_name, device=0 if torch.cuda.is_available() else -1
)


def translate_between_languages(text, model_name):

    # Count the number of tokens in the input text
    num_tokens = len(tokenizer(text)["input_ids"])
    translated = translator(text, src_lang="en", tgt_lang="sv")
    return translated[0]["translation_text"]


def translate_english_to_swedish(english_text):
    swedish_text = translate_between_languages(
        english_text, "Helsinki-NLP/opus-mt-en-sv"
    )
    print(
        colored("The Swedish translation is:", "green"), colored(swedish_text, "yellow")
    )
    return swedish_text


def translate_jsonl(input_file, output_file, corrupt_lines_file):
    with jsonlines.open(input_file) as reader, jsonlines.open(
        output_file, mode="w"
    ) as writer:
        # Loop through each line of the input file
        for line in reader:
            text = line["text"]
            source = line["meta"]["source"]
            # Count the number of tokens in the input text
            num_tokens = len(tokenizer(text)["input_ids"])
            if num_tokens > tokenizer.model_max_length:
                print(
                    colored(
                        f"The input text is too long: {num_tokens} tokens (max allowed: ({tokenizer.model_max_length})",
                        "red",
                    )
                )
                with open(corrupt_lines_file, "a") as fc:
                    fc.write(str(line) + "\n")
                continue
            dialogue_pairs = text.split("User:")

            thread = []
            for dialogue_pair in dialogue_pairs:
                if (
                    dialogue_pair == ""
                ):  # the first occurrence of User: is removed by the split
                    continue
                try:
                    user_text = dialogue_pair.split("\nAssistant:")[0]
                    assistant_response = dialogue_pair.split("\nAssistant:")[1]
                    print(
                        colored("\nThe user text is:", "green"),
                        colored(user_text, "blue"),
                    )
                except:
                    print(colored("The line is corrupt, skipping", "red"))
                    with open(corrupt_lines_file, "a") as fc:
                        fc.write(str(line) + "\n")
                # Translate the user's text and the assistant's response
                user_text_translated = translate_english_to_swedish(user_text)
                print(
                    colored("The assistant response is:", "green"),
                    colored(assistant_response, "red"),
                )
                assistant_response_translated = translate_english_to_swedish(
                    assistant_response
                )

                # Add the translated user text and assistant response to the thread
                thread.append({"text": user_text_translated, "role": "prompter"})
                thread.append(
                    {"text": assistant_response_translated, "role": "assistant"}
                )

            new_line = {
                "thread": thread,
                "source": source,  # Add the source to the new line
            }
            print(
                colored("The translated line is:", "green"),
                colored(new_line, "magenta"),
            )
            print(colored("\n******************\n", "black"))
            # Write the new line to the output file
            writer.write(new_line)


if __name__ == "__main__":
    translate_jsonl(
        "oa_v3_fixed_plus_safety.jsonl",
        "LAION-multilingual-bot-swedish.jsonl",
        "corrupt_lines.txt",
    )
