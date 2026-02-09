def format_chat_history(chat_history, max_turns=6):
    return "\n".join(chat_history[-max_turns:])


def update_memory(chat_history, user, assistant):
    chat_history.append(f"User: {user}")
    chat_history.append(f"Assistant: {assistant}")
