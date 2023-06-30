from chatbot import Chatbot

# Initialize chatbot
chat = Chatbot()
print("user: ", end = "")
while True:
    prompt = input()
    past = [""]
    for token in chat.chat(prompt):
        past[-1] += token
        if "." in token:
            print(past[-1], end = "")
            past.append("")
    print()