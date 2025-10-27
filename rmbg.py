import onnxruntime as ort
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from utilities import postprocess_image, preprocess_image


def initialize_model(model_path):
    ort_sess = ort.InferenceSession(model_path)
    return ort_sess


def process_image_buffer(image_buffer, ort_sess):
    image = Image.open(BytesIO(image_buffer))
    image_np = np.array(image)
    orig_im_size = image_np.shape[0:2]

    model_input_size = [1024, 1024]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_image = preprocess_image(image_np, model_input_size).to(device)

    image_np_for_model = processed_image.cpu().numpy(
    ) if device.type == 'cuda' else processed_image.numpy()

    input_name = ort_sess.get_inputs()[0].name
    result = ort_sess.run(None, {input_name: image_np_for_model})

    result_tensor = torch.tensor(result[0][0], dtype=torch.float32)
    result_image = postprocess_image(result_tensor, orig_im_size)

    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    no_bg_image.paste(image, mask=pil_im)

    output_buffer = BytesIO()
    no_bg_image.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    return output_buffer.getvalue()


model_path = "model/model_fp16.onnx"
ort_sess = initialize_model(model_path)


# if __name__ == "__main__":
#     test()
