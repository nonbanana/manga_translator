from google.cloud import vision


def detect_text(image_path, auth_path):
    """
    Detects text in the file.
    https://cloud.google.com/vision/docs/ocr#vision_text_detection-python

    :params input_image: image path
    :params auth_path: google api auth file path
    """
    client = vision.ImageAnnotatorClient.from_service_account_json(auth_path)

    # with io.open(path, 'rb') as image_file:
    #     content = image_file.read()
    with open(image_path, 'rb') as input_image:
        image = vision.Image(content=input_image.read())

    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return response

    
if __name__ == "__main__":
    ocr_image = "./manga.png"
    ocr_result = detect_text(ocr_image, '/home/xecachan/studies/manga/api-project-46802515283-0b4891073f4f.json')
    print(ocr_result)
