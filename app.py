import gradio as gr
from transformers import pipeline

# Load a model from Hugging Face Hub
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Gradio UI
demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Enter text"),
    outputs=gr.Textbox(label="Summary"),
    title="Hugging Face Summarizer"
)

if __name__ == "__main__":
    demo.launch()
