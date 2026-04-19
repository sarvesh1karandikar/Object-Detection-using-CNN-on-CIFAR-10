import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
CLASS_EMOJIS = {'airplane':'✈️','automobile':'🚗','bird':'🐦','cat':'🐱','deer':'🦌','dog':'🐶','frog':'🐸','horse':'🐴','ship':'🚢','truck':'🚛'}

model = tf.keras.models.load_model('cifar10_model.keras')

def classify(image):
    img = Image.fromarray(image).resize((32, 32))
    x = np.array(img).astype('float32') / 255.0
    x = np.expand_dims(x, 0)
    preds = model.predict(x, verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]
    results = {f"{CLASS_EMOJIS[CLASSES[i]]} {CLASSES[i]}": float(preds[i]) for i in top3}
    return results

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="CIFAR-10 Image Classifier",
    description="Upload any image — the model will classify it into one of 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Trained to ~75%+ accuracy.",
    examples=[],  # can be populated manually later
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
