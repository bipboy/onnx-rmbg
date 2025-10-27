from skimage import io
import onnxruntime as ort
import numpy as np
import torch
import os
from PIL import Image

from utilities import postprocess_image, preprocess_image


def test():
    ort_sess = ort.InferenceSession("model/model_fp16.onnx")
    im_path = f"{os.path.dirname(os.path.abspath(__file__))}/example_input.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_input_size = [1024, 1024]
    orig_im = io.imread(im_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    image_np = image.cpu().numpy() if device.type == 'cuda' else image.numpy()
    input_name = ort_sess.get_inputs()[0].name
    result = ort_sess.run(None, {input_name: image_np})

    result_tensor = torch.tensor(result[0][0], dtype=torch.float32)
    result_image = postprocess_image(result_tensor, orig_im_size)

    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(im_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save("000001.png")


if __name__ == "__main__":
    test()
