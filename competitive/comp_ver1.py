import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import re
from datasets import load_dataset
from transformers import NougatProcessor,VisionEncoderDecoderModel
import torch

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

filepath = hf_hub_download(repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_paper.png", repo_type="dataset")
filepath = "Dataset/HandwrittenData/" + "images/test/" + "00B9DrMf1.png"
image = Image.open(filepath)
image = image.convert("RGB")
image.show()
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

outputs = model.generate(pixel_values=pixel_values, min_length=1,
    max_new_tokens=30,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
# note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
print(sequence)

