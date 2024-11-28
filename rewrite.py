# Define a function to convert each line to natural language
def convert_to_natural_language(line):
    parts = line.strip().split(',')
    
    # Determine the speaker's gender
    if parts[0] == '<sp:feminine>':
        speaker = "I am a woman."
    elif parts[0] == '<sp:masculine>':
        speaker = "I am a man."
    else:
        speaker = ""
    
    # Check if there is interlocutor information
    if any(parts[1:]):
        # Determine the formality of the conversation
        if '<informal>' in parts:
            formality = "I am having an informal chat with"
        elif '<formal>' in parts:
            if not parts[1] and not parts[2]:
                formality = "I am having a formal conversation."
            else:
                formality = "I am having a formal conversation with"
        else:
            formality = "I am talking to"
        
        # Determine the interlocutor's gender and number
        if '<il:masculine>' in parts and '<singular>' in parts:
            interlocutor = " a man."
        elif '<il:feminine>' in parts and '<singular>' in parts:
            interlocutor = " a woman."
        elif '<il:masculine>' in parts and '<plural>' in parts:
            interlocutor = " a group of men."
        elif '<il:feminine>' in parts and '<plural>' in parts:
            interlocutor = " a group of women."
        elif '<il:mixed>' in parts and '<plural>' in parts:
            interlocutor = " a group of people."
        else:
            if '<singular>' in parts:
                interlocutor = " a person."
            elif '<plural>' in parts:
                interlocutor = " a group of people."
            else:
                interlocutor = ""
        
        # Combine the sentences
        return f"{speaker} {formality}{interlocutor}"
    else:
        if speaker:
            return speaker
        elif '<formal>' in parts:
            return "I am having a formal conversation."

# Read the content of the file
with open('data_retired/dev.cxt', 'r') as file:
    lines = file.read().splitlines()

# Convert each line to natural language
converted_lines = [convert_to_natural_language(line).strip() for line in lines]

with open('data_retired/dev.context', 'w+') as file:
    for line in converted_lines:
        file.write(line + '\n')