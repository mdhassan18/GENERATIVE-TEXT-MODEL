# GENERATIVE-TEXT-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MOHAMMED HASSAN M

*INTERN ID*: CT04DG1935

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

# 📄 Project Description: Text Generation Using GPT-2 and LSTM
In this project, I implemented a dual-model text generation system that allows users to generate coherent and contextually relevant paragraphs using either a pre-trained transformer model (GPT-2) or a custom-trained recurrent neural network (LSTM). The aim of the task was to explore and demonstrate the capabilities of modern natural language generation (NLG) techniques using both state-of-the-art pretrained models and foundational deep learning architectures. The final deliverable is a Python script that enables users to interactively select the model they wish to use and input a custom topic or prompt for generating textual content.

The first part of the system utilizes GPT-2, a transformer-based language model developed by OpenAI. It is a pre-trained model capable of generating highly fluent and contextually accurate paragraphs when given a text prompt. In this setup, I integrated the GPT-2 model using the Hugging Face transformers library. The code loads the tokenizer and model, encodes the input prompt, and generates a sequence of text tokens using sampling techniques such as top-k sampling, top-p (nucleus) filtering, and temperature adjustment to make the text more diverse and human-like. GPT-2 is particularly suitable for real-world use cases such as content creation, conversational AI, writing assistants, and more.

The second part of the system is a simple yet educational character-level LSTM model built using PyTorch. This model is trained locally on a short custom dataset — a dummy text string that is repeated and formatted to suit a sequence prediction task. The LSTM (Long Short-Term Memory) network learns character-to-character transitions and, after training, is capable of generating a stream of characters that form readable words and sentences. Although this model doesn’t reach the quality of GPT-2, it is useful for educational purposes and helps illustrate how basic recurrent networks can be used to learn language structure from scratch.

To facilitate user interaction, the script offers a command-line interface where the user can choose between the two models. Upon selection, the program prompts the user to input a topic or seed text. The corresponding model then generates and outputs a paragraph based on that input. This design allows users to directly compare the output of a pre-trained large language model and a simple LSTM trained locally.

# Tools and Libraries Used:

Python 3.10: The programming language used for building the entire project.

PyTorch: Used for implementing and training the LSTM model.

Transformers (Hugging Face): For loading and using the GPT-2 model.

VS Code & Command Prompt: For writing, running, and testing the script.

random, torch.nn.functional: For sampling and managing predictions in the LSTM model.

# Applications of This Project:

This project simulates the core functionality found in advanced content-generation platforms and AI-powered writing tools. It can be applied in various fields, including:

Creative writing: Generating story plots, dialogues, or poetry.

Customer support: Auto-generating responses based on common queries.

Education: Demonstrating the difference between simple and advanced text generation models.

Marketing: Creating product descriptions, social media captions, or ad copy.

AI research: Comparing RNN vs transformer models for sequence generation tasks.

In conclusion, the task involved building a hybrid text generation tool that leverages the strengths of both classical and modern AI techniques. Through this project, I gained hands-on experience in NLP, deep learning, model integration, and practical Python programming. This system not only demonstrates the evolution of language modeling architectures — from RNN-based models like LSTM to transformers like GPT-2 — but also provides a flexible interface for experimenting with text generation in real time.

<img width="1919" height="1079" alt="Image" src="https://github.com/user-attachments/assets/b152e187-ba12-4741-aba6-c0805741effa" />
