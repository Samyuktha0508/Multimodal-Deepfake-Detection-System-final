import gradio as gr
import pipeline
import numpy as np
import librosa

# =========================
# AUDIO PROCESSING (ONLY ON CLICK)
# =========================
def process_audio(audio):
    try:
        import numpy as np

        # 🔥 CASE 1: Proper (x, sr)
        if isinstance(audio, tuple) and len(audio) == 2:
            x, sr = audio

        # 🔥 CASE 2: weird scalar / broken input
        elif isinstance(audio, (float, np.float32, np.float64)):
            return "Invalid audio input"

        else:
            return "Unsupported audio format"

        x = np.array(x)

        # 🔥 FIX: if scalar sneaks inside
        if x.ndim == 0:
            return "Invalid audio data"

        # stereo → mono
        if x.ndim > 1:
            x = np.mean(x, axis=1)

        # normalize
        x = x / (np.max(np.abs(x)) + 1e-6)
        x = x.astype(np.float32)

        # model requirement
        target_len = 64600

        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)))
        else:
            x = x[:target_len]

        return (x, sr)

    except Exception as e:
        return str(e)

# =========================
# MAIN FUNCTION
# =========================
def predict(input_type, image, video, audio):

    try:
        if input_type == "Image":
            if image is None:
                return "Upload an image"
            return pipeline.deepfakes_image_predict(image)

        elif input_type == "Video":
            if video is None:
                return "Upload a video"
            return pipeline.deepfakes_video_predict(video)

        elif input_type == "Audio":
            if audio is None:
                return "Upload audio"

            processed = process_audio(audio)

            if isinstance(processed, str):
                return f"Audio Error: {processed}"

            return pipeline.deepfakes_audio_predict(processed)

        else:
            return "Invalid input"

    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# CLEAR FUNCTION
# =========================
def clear_all():
    return "Image", None, None, None, ""


# =========================
# UI
# =========================
with gr.Blocks() as app:

    gr.Markdown("# 🔥 Multimodal Deepfake Detector")
    gr.Markdown("Detect fake Images, Videos, and Audio using AI")

    input_type = gr.Radio(
        ["Image", "Video", "Audio"],
        value="Image",
        label="Select Input Type"
    )

    image_input = gr.Image(label="Upload Image")
    video_input = gr.Video(label="Upload Video")
    audio_input = gr.Audio(type="numpy", label="Upload Audio")

# Set visibility after creation
    image_input.visible = True
    video_input.visible = False
    audio_input.visible = False
    output = gr.Textbox(label="Result", lines=3)

    submit = gr.Button("Analyze")
    clear = gr.Button("Clear")

    # =========================
    # SWITCH INPUTS
    # =========================
    def update(choice):
        return (
            gr.update(visible=choice == "Image"),
            gr.update(visible=choice == "Video"),
            gr.update(visible=choice == "Audio"),
        )

    input_type.change(
        update,
        inputs=input_type,
        outputs=[image_input, video_input, audio_input]
    )

    # =========================
    # ACTIONS
    # =========================
    submit.click(
        predict,
        inputs=[input_type, image_input, video_input, audio_input],
        outputs=output
    )

    clear.click(
        clear_all,
        outputs=[input_type, image_input, video_input, audio_input, output]
    )


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.queue().launch()