
import json
import glob
import os.path
import pickle

from PIL import ImageFont, Image

import img_tool
import script_extractor as scriptext


if __name__ == "__main__":
    # input_path = './百合乱暴される淫魔01/*'
    input_path = './偕老同穴の契り/*'
    font_src = './NotoSansCJKkr-Regular (TTF).ttf'
    gc_auth = './google_api_key.json'
    image_paths = glob.glob(input_path + '.webp')
    image_paths = sorted(list(image_paths))
    # image_paths += glob.glob(input_path + '.png')
    for image_path in image_paths:
        if 'result' in image_path:
            continue
        
        img = Image.open(image_path)
        fontsize = img.width // 35
        font = ImageFont.truetype(font_src, fontsize)

        json_path = image_path + '.ocr.json'
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf8') as fp:
                script = json.load(fp)
        else:
            # pickle_path = image_path + '.pickle'
            # if os.path.exists(pickle_path):
            #     # pickle 있으면
            #     with open(pickle_path, 'rb') as fp:
            #         ocr_response = pickle.load(fp)
            # else:
            #     # pickle 없으면 새로 OCR
            #     ocr_response = scriptext.doGoogleOCR(image_path, gc_auth)
            #     with open(pickle_path, 'wb') as fp:
            #         pickle.dump(ocr_response, fp)
            
            script = scriptext.do_paddlepaddle(image_path)
            # word_data= scriptext.cvtGoogleOCRToRawData(ocr_response)        
            # script = scriptext.makeScript(word_data)
            scriptext.translateScript(script, service="google")
            with open(json_path, 'w', encoding='utf8') as fp:
                json.dump(script, fp, ensure_ascii=False, indent=4)

        img_tool.drawOCRCluster(img, script, font)

        img.save(image_path + '.result.jpg')
