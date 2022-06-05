from PIL import ImageFont, ImageDraw, Image
import json
from sklearn.cluster import DBSCAN
import pickle
import gcloud_ocr
import glob
import os.path
import translators as ts

def drawOCRText(input_img, ocr_response, font):
    """
    draw text from ocr response
    :img_src: PIL 이미지 소스
    :ocr_response: OCR 이미지 결과
    """
    draw = ImageDraw.Draw(input_img)
    text_annotation = ocr_response.text_annotations
    # text_annotation = ocr_response["responses"][0]['textAnnotations'][1:]
    for text_block in text_annotation[1:]:
        print(text_block)


        xpos_list = []
        ypos_list = []
        for pos in text_block.bounding_poly.vertices:
            xpos_list.append(pos.x)
            ypos_list.append(pos.y)
        ypos_min = min(ypos_list)
        ypos_max = max(ypos_list)
        xpos_min = min(xpos_list)
        xpos_max = max(xpos_list)
        
        # 글자 지우기
        draw.rectangle((xpos_min, ypos_min, xpos_max, ypos_max), fill=256)
        ### 번역 및 글자 장평 너비 조절
        # 글자 그리기
        draw.text((xpos_min, ypos_min), text_block.description, font=font, fill=(0))
        # print(height, width)

from manga_ocr import MangaOcr
def doMangaOCR(image_path):
    mocr = MangaOcr()
    ocr_result = mocr(image_path)
    return ocr_result

def doGoogleOCR(image_path, gc_auth):
    ocr_result = gcloud_ocr.detect_text(open(image_path, 'rb'), gc_auth)
    return ocr_result
    
def cvtGoogleOCRToRawData(ocr_response):
    text_annotation = ocr_response.text_annotations
    ocr_result_raw = []
    # 좌표(?) 만들기 
    text_pos = []
    for text_block in text_annotation[1:]:
        xpos_list = []
        ypos_list = []
        for pos in text_block.bounding_poly.vertices:
            xpos_list.append(pos.x)
            ypos_list.append(pos.y)
        ocr_result_raw.append({
        "description" : text_block.description, 
        "pos" : {
            "x_min" : min(xpos_list), 
            "x_max" : max(xpos_list), 
            "y_min" : min(ypos_list), 
            "y_max" : max(ypos_list)
            }
        })
    return ocr_result_raw

def makeScript(word_data):
    # 클러스터링을 위해 좌표 노드(?) 만들기 
    text_pos = []
    for word in word_data:
        text_pos += [[word["pos"]["y_min"], word["pos"]["x_min"]],
                     [word["pos"]["y_min"], word["pos"]["x_max"]],
                     [word["pos"]["y_max"], word["pos"]["x_min"]],
                     [word["pos"]["y_max"], word["pos"]["x_max"]]]
        
    db_scan = DBSCAN(eps=100, min_samples=3).fit(text_pos)

    script_cluster = [{
                "script" : "",
                "trans_script" : "",
                "pos" : {"x_min" : 0, "x_max" : 0, "y_min" : 0, "y_max" : 0}, 
                "raw" : []
            } 
            for _ in range(max(db_scan.labels_) + 2)
        ]

    for i, text_block in enumerate(word_data):
        # bounding_poly.vertices.pos.y
        label = db_scan.labels_[i*4+2] + 1
        script_cluster[label]["raw"].append(text_block)

    for script in script_cluster:
        if not script["raw"]:
            continue
        script["raw"].sort(key=lambda x: x["pos"]["y_min"])
        script["raw"].sort(key=lambda x: x["pos"]["x_max"], reverse=True)
        xpos_list = []
        ypos_list = []
        for script_raw in script["raw"]:
            script["script"] += script_raw["description"] 
            xpos_list.append(script_raw["pos"]["x_min"])
            ypos_list.append(script_raw["pos"]["y_min"])
            xpos_list.append(script_raw["pos"]["x_max"])
            ypos_list.append(script_raw["pos"]["y_max"])
        script["pos"]["x_min"] = min(xpos_list)
        script["pos"]["x_max"] = max(xpos_list)
        script["pos"]["y_min"] = min(ypos_list)
        script["pos"]["y_max"] = max(ypos_list)
    
    return script_cluster



def drawOCRCluster(input_img, script_cluster, font):
    """
    draw text from ocr response
    :img_src: PIL 이미지 소스
    :ocr_response: OCR 이미지 결과
    """
    draw = ImageDraw.Draw(input_img)

    for script in script_cluster:
        # 글자 지우기
        draw.rectangle((script["pos"]["x_min"], script["pos"]["y_min"], script["pos"]["x_max"], script["pos"]["y_max"]), fill=255)
        ### 번역 및 글자 장평 너비 조절
        # 글자 그리기
        translated_script  = script["trans_script"] .replace(' ', '\n')
        draw.multiline_text((script["pos"]["x_min"], script["pos"]["y_min"]), translated_script, font=font, fill=(0))

def translateOCR(script_cluster):
    for script in script_cluster:
        script["trans_script"] = ts.papago(script["script"], from_language="ja", to_language="ko")

input_path = './sample/*'
font_src = './NotoSansCJKkr-Regular (TTF).ttf'
gc_auth = './google_api_key.json'
image_paths = glob.glob(input_path + '.jpg')
# image_paths += glob.glob(input_path + '.png')
for image_path in image_paths:
    if 'result' in image_path:
        continue
    font = ImageFont.truetype(font_src, 30)
    img = Image.open(image_path)

    # from_pickle = True
    json_path = image_path + '.ocr.json'
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf8') as fp:
            script = json.load(fp)
    else:
        pickle_path = image_path + '.pickle'
        if os.path.exists(pickle_path):
            # pickle 있으면
            with open(pickle_path, 'rb') as fp:
                ocr_response = pickle.load(fp)
        else:
            # pickle 없으면
            ocr_response = doGoogleOCR(image_path, gc_auth)
            with open(pickle_path, 'wb') as fp:
                pickle.dump(ocr_response, fp)
        
        word_data= cvtGoogleOCRToRawData(ocr_response)        
        script = makeScript(word_data)
        translateOCR(script)
        with open(json_path, 'w', encoding='utf8') as fp:
            json.dump(script, fp, ensure_ascii=False, indent=4)

    drawOCRCluster(img, script, font)

    img.save(image_path + '.result.jpg')
