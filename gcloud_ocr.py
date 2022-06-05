# from PIL import Image
# from io import FileIO

# import requests
# import base64
# import json
# import io
# def ocr(input_image, gc_auth_json):
#     """
#     image: (file object)
#     """
#     url = "https://vision.googleapis.com/v1/images:annotate"
#     image_base64 = base64.b64encode(input_image.read())
#     # print(image_base64)
#     data = {
#         "requests": [{
#             "image": {"content": str(image_base64)},
#             "features": [{"type": "TEXT_DETECTION"}]
#         }]
#     }
#     headers = {
#         # "Authorization": "Bearer " + gc_auth_json.read(), 
#         "Content-Type": "application/json; charset=utf-8"
#     }
#     # print(json.dumps(data))
#     res = requests.post(url, headers=headers, data=json.dumps(data) , auth=requests.BearerAuth())
#     return res

def detect_text(input_image, auth_path):
    """
    Detects text in the file.
    https://cloud.google.com/vision/docs/ocr#vision_text_detection-python
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient.from_service_account_json(auth_path)

    # with io.open(path, 'rb') as image_file:
    #     content = image_file.read()

    image = vision.Image(content=input_image.read())

    response = client.text_detection(image=image)
    # texts = response.text_annotations
    # print('Texts:')

    # for text in texts:
    #     print('\n"{}"'.format(text.description))

    #     vertices = (['({},{})'.format(vertex.x, vertex.y)
    #                 for vertex in text.bounding_poly.vertices])

    #     print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return response
if __name__ == "__main__":
    ocr_image = open("./manga.png", 'rb')
    ocr_result = detect_text(ocr_image, '/home/xecachan/studies/manga/api-project-46802515283-0b4891073f4f.json')
    print(ocr_result)
