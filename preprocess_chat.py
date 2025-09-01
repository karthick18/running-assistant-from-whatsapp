#!/usr/bin/env python3

# This file preprocesses the whatsapp chat export text file to a format suitable for creating a vector store.
# It removes unnecessary lines, timestamps, and formats the text.
import re
import json

MIN_MESSAGE_LENGTH = 20
MIN_MESSAGES_DAILY_COUNT = 3

def message_is_omitted(message: str) -> bool:
    omitted_phrases = ['image omitted', 'video omitted', 'sticker omitted', 'gif omitted', 'audio omitted', 'document omitted']
    omitted: bool = any(message.endswith(phrase) for phrase in omitted_phrases)
    if omitted:
        return True
    # check for deleted messages
    if message.find('this message was deleted') != -1:
        return True

# group messages by date
def preprocess_chat_by_date(input_file: str, output_file: str) -> None:
    preprocessed_contents = []
    last_date = None
    messages_by_date = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i < 100:
                continue
            # Remove lines that do not contain messages (e.g., "Messages to this chat and calls are now secured with end-to-end encryption.")
            matches = re.match(r'.*?\[(\d{1,2}/\d{1,2}/\d{1,2}).*?[AM|PM]\]\s+(.+?): (.+)', line)
            if matches:
                date, sender, message = matches.groups()
                message = message.lower()
                if message_is_omitted(message):
                    continue
                # remove <this message was edited> from the message
                message = message.replace('<this message was edited>', '')
                if len(message) < MIN_MESSAGE_LENGTH:
                    continue
                if last_date is None or date != last_date:
                    # group the messages by sender to the same date in a list
                    if last_date is not None and len(messages_by_date) >= MIN_MESSAGES_DAILY_COUNT:
                        preprocessed_contents.append({'date':last_date, 'messages': messages_by_date})
                    last_date = date
                    messages_by_date = []
                    messages_by_date.append({ "sender": sender, "message": message })
                else:
                    # same date, append the message to the last sender if same sender else create a new entry
                    last_sender = messages_by_date[-1]['sender']
                    if sender == last_sender:
                            messages_by_date[-1]['message'] += ' ' + message
                    else:
                        messages_by_date.append({ "sender": sender, "message": message })

    # append the last group of messages
    if last_date is not None and len(messages_by_date) >= MIN_MESSAGES_DAILY_COUNT:
        preprocessed_contents.append({'date':last_date, 'messages': messages_by_date})

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(preprocessed_contents, outfile, ensure_ascii=False, indent=4)

def preprocess_chat(input_file: str, output_file: str) -> None:
    preprocessed_contents = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # Remove lines that do not contain messages (e.g., "Messages to this chat and calls are now secured with end-to-end encryption.")
            matches = re.match(r'(.*?\[\d{1,2}/\d{1,2}/\d{1,2}.*?[AM|PM]\]).*?(.+?): (.+)', line)
            if matches:
                _, sender, message = matches.groups()
                message = message.lower()
                if message_is_omitted(message):
                    continue
                data = { "sender": sender, "message": message }
                preprocessed_contents.append(data)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(preprocessed_contents, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess WhatsApp chat export text file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-file", type=str, default='running_chats.txt', help="The path to the input WhatsApp chat export text file.")
    parser.add_argument("--output-file", type=str, default='running_chats_processed.txt', help="The path to the output preprocessed text file.")
    parser.add_argument("--by-date", action='store_true', help="Preprocess the chat by grouping messages by date.")
    args = parser.parse_args()
    if args.by_date:
        print("Preprocessing chat by date...")
        preprocess_chat_by_date(args.input_file, args.output_file)
    else:
        preprocess_chat(args.by_date, args.input_file, args.output_file)