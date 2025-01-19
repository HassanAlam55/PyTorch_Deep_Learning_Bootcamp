# 1. import and setup
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_repeat as timer  
from typing import Tuple, Dict


# 2. Models and Transforms
effnetb2, effnetb2_transforms = create_effnetb2_model(output_feature_count=3)

effnetb2.load_state_dict(
    torch.load(
        # f=f"models/09_pre_trained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent_{10}_epochs.pth",
        f=f"09_pre_trained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent_{10}_epochs.pth",        
        map_location=torch.device('cpu')
    )
)

# 3. Predict Funciton
# def predict(img) -> Tuple[Dict, float]
# recreate workflow
def predict(img) ->Tuple [Dict, float]:

    start_time = timer()

    img = effnetb2_transforms(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim = 1)

    pred_labels_probs = {class_names[i]:float (pred_probs[0][i]) for i in range (len(class_names))}

    pred_time = round (timer() - start_time , 5)

    return pred_labels_probs, pred_time

#  Gradio Ap
title = "Food vision Mini 🍕🥩🍣"
description = "an EfficentNetB2 feature extract"
article = "Created at Hassan 09"

# these will likely cause problems in relative paths. Need to fix. 
# example_list = [[str(filepath)] for filepath in random.sample (test_data_paths, k = 3)]
example_list = [['examples/' + example] for example in os.listdir('examples')]

demo = gr.Interface(fn=predict,
                    inputs = gr.Image(type='pil'),
                    outputs = [gr.Label(num_top_classes = 3, label = "Prdictions"),
                                gr.Number(label = "prediction time")],
                    examples = example_list,
                    title = title,
                    description=description,
                    article=article)

demo.launch(debug=False,
           share=True)
