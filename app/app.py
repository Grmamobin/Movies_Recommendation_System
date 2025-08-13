from tensorflow.keras.models import load_model
import gradio as gr

# Load your trained recommender model
model = load_model("/Users/macbookpro/Recommander.h5")

# Example function: takes user input and returns top-k titles + poster URLs
def recommend_movies(user_input):
    # Replace with your actual model inference
    # Example:
    top_titles = ["Movie A", "Movie B", "Movie C"]
    top_posters = ["url_to_image_A", "url_to_image_B", "url_to_image_C"]
    # For demo, we just return the first movie poster and title
    return top_posters[0], top_titles[0]

with gr.Blocks() as demo:
    gr.Markdown("Movie Recommender")
    
    with gr.Row(equal_height=True):
        textbox = gr.Textbox(lines=1, placeholder="Enter user ID or movie title...", show_label=False)
        button = gr.Button("Recommender", variant="primary")
    
    output_image = gr.Image(height=200)
    output_text = gr.Textbox(label="Recommended Movie")
    
    button.click(
        fn=recommend_movies,
        inputs=textbox,
        outputs=[output_image, output_text]
    )

demo.launch(share=True)
