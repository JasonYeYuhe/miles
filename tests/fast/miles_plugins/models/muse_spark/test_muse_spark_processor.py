import base64
import io

from PIL import Image

from miles_plugins.models.muse_spark.vision.processor import _load_pil_image


def test_data_uri_decodes_to_the_encoded_image():
    image = Image.new("RGB", (5, 3), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    # chenhegu/geo3k_imgurl stores its images as data:image/None;base64,... (the mime type is unset).
    uri = "data:image/None;base64," + base64.b64encode(buffer.getvalue()).decode()
    decoded = _load_pil_image(uri)
    assert decoded.size == (5, 3) and decoded.convert("RGB").getpixel((0, 0)) == (10, 20, 30)
