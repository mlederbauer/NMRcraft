import base64

import gradio as gr


# Function to encode the image to a base64 data URI
def image_to_data_uri(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"


# Encode your logo
image_data_uri = image_to_data_uri("assets/NMRcraft-logo.png")

# HTML image tag with the base64-encoded image
logo_html = f"<img src='{image_data_uri}' alt='NMRcraft Logo' style='max-width:100%; height: auto; display: block; margin-left: auto; margin-right: auto;'/>"


# Define a placeholder response function
def respond(input_text):
    return f"You said: {input_text}"


# Set up Gradio interface
with gr.Blocks() as demo:
    # Display the logo at the top of the interface
    gr.Markdown(logo_html)

    # Your interface components
    with gr.Row():
        input_text = gr.Textbox(label="Enter your text below:")
        output_text = gr.Textbox(label="Response")
        input_text.change(fn=respond, inputs=input_text, outputs=output_text)

# Launch the interface
demo.launch()
