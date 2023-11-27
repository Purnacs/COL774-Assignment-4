# %% [markdown]
# <a href="https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Set-up environment
# 
# We install 🤗 Transformers, which includes many awesome models, including TrOCR by Microsoft Research.

# %%
# !pip install -q transformers

# %% [markdown]
# ## Load image
# 
# Here we load an image from the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), on which TrOCR was fine-tuned.

# %%
import requests
from PIL import Image

url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image

# %% [markdown]
# ## Prepare image for model
# 
# Next, we use `TrOCRProcessor` (which wraps a `ViTFeatureExtractor` and a `RobertaTokenizer` into one) to resize + normalize the image for the model.

# %%
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# calling the processor is equivalent to calling the feature extractor
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

# %% [markdown]
# ## Load model
# 
# Here we load a TrOCR model from the [hub](https://huggingface.co/models?other=trocr). TrOCR models are instances of [`VisionEncoderDecoderModel`](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder), which combine a vision encoder (like ViT, BEiT, DeiT, ...) with a language model as decoder (like BERT, RoBERTa, GPT-2, ...). 

# %%
from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# %% [markdown]
# ## Generate text 
# 
# Finally, we can generate text autoregressively using the `.generate()` method. We use the tokenizer part of the `processor` to decode the generated id's back to text. Note that by default, greedy search is used for generation, but there are more fancy methods like beam search and top-k sampling, which are also supported. You can check out [this blog post](https://huggingface.co/blog/how-to-generate) for details.

# %%
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# %% [markdown]
# ## Gradio demo
# 
# We can also wrap up everything we wrote above in a quick [Gradio](https://gradio.app/) demo. Gradio is an awesome tool for creating quick ML demos!
# 
# We just need to define a function that takes in an image input and produces a text output (note that Gradio supports many other input & output formats)!

# %%
# !pip install -q gradio

# %%
# # import gradio as gr
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import requests
# from PIL import Image

# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# # load image examples from the IAM database
# urls = ['https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoolxi9yWGAT5SLZShv8vVd0bz47UWRzQC19fDTeE8GmGv_Rn-PCF1pP1rrUx8kOjA4gg&usqp=CAU',
#         'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRNYtTuSBpZPV_nkBYPMFwVVD9asZOPgHww4epu9EqWgDmXW--sE2o8og40ZfDGo87j5w&usqp=CAU']
# for idx, url in enumerate(urls):
#   image = Image.open(requests.get(url, stream=True).raw)
#   image.save(f"image_{idx}.png")

# def process_image(image):
#     # prepare image
#     pixel_values = processor(image, return_tensors="pt").pixel_values

#     # generate (no beam search)
#     generated_ids = model.generate(pixel_values)

#     # decode
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     return generated_text

# title = "Interactive demo: TrOCR"
# description = "Demo for Microsoft's TrOCR, an encoder-decoder model consisting of an image Transformer encoder and a text Transformer decoder for state-of-the-art optical character recognition (OCR) on single-text line images. This particular model is fine-tuned on IAM, a dataset of annotated handwritten images. To use it, simply upload an image or use the example image below and click 'submit'. Results will show up in a few seconds."
# article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2109.10282'>TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models</a> | <a href='https://github.com/microsoft/unilm/tree/master/trocr'>Github Repo</a></p>"
# examples =[["image_0.png"], ["image_1.png"], ["image_2.png"]]

# iface = gr.Interface(fn=process_image, 
#                      inputs=gr.inputs.Image(type="pil"), 
#                      outputs=gr.outputs.Textbox(),
#                      title=title,
#                      description=description,
#                      article=article,
#                      examples=examples)
# iface.launch(debug=True)

# # %%



