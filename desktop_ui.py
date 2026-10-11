import gradio as gr
import torch
from PIL import Image
from src import config, utils, preprocess, model as model_loader


class_to_idx, idx_to_class = utils.load_class_mapping()
num_classes = len(idx_to_class)
model, _, _, _ = utils.load_model(config.MODEL_PATH_FOR_APP, num_classes_override=num_classes)
model.eval()
model.to(config.DEVICE)

def predict_gradio_interface(image_pil):
    if image_pil is None:
        return "Görsel bulunamadı", 0.0

    transform = preprocess.get_val_test_transforms()
    img_tensor = transform(image_pil).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class_name = idx_to_class.get(predicted_idx.item(), "Bilinmiyor")
    confidence_score = round(confidence.item() * 100, 2)

    return f"🌸 Tahmin: {predicted_class_name}", confidence_score

# Gradio Arayüzü
with gr.Blocks(theme=gr.themes.Soft(), css="""
.gradio-container {background: #fff0f5;}
#title {color: #c71585; font-size: 28px; text-align: center; font-weight: bold;}
#description {color: #8b008b; font-size: 16px; text-align: center;}
#predict-btn {background-color: #ff69b4;}
#clear-btn {background-color: #ffc0cb;}
""") as app:

    gr.Markdown("## <div id='title'>🌺 MultiZoo Hayvan Tanıma Sistemi</div>")
    gr.Markdown("<div id='description'>Bir hayvan görseli yükleyin, model türünü tahmin etsin ve güven skoru versin.</div>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="🐾 Hayvan Görseli", elem_id="img-input")
            predict_button = gr.Button("🚀 Tahmini Yap", elem_id="predict-btn")
            clear_button = gr.Button("🧹 Temizle", elem_id="clear-btn")
        with gr.Column(scale=1):
            label_output = gr.Textbox(label="🌼 Tahmin Sonucu", elem_id="output-label")
            score_output = gr.Number(label="💖 Güven Skoru (%)", value=0, elem_id="output-score")

    predict_button.click(fn=predict_gradio_interface, inputs=image_input, outputs=[label_output, score_output])
    clear_button.click(lambda: (None, "", 0), outputs=[image_input, label_output, score_output])

if __name__ == "__main__":
    app.launch()
