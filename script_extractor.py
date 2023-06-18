import gcloud_ocr
from sklearn.cluster import DBSCAN
import translators as ts
from PIL import Image
import numpy as np


# TODO - manga_ocr 테스트 필요
from manga_ocr import MangaOcr
def doMangaOCR(mocr, image_path):
    
    ocr_result = mocr(image_path)
    return ocr_result

def do_paddlepaddle(image_path):
    from paddleocr import PaddleOCR,draw_ocr
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.
    ocr = PaddleOCR(use_angle_cls=True, lang='japan') # need to run only once to download and load model into memory
    # img_path = './imgs_en/img_12.jpg'
    img = np.array(Image.open(image_path))
    result = []
    crop_boxes = [
        [0,0,img.shape[0]*3//5,img.shape[1]*3//5],
        [img.shape[0]*2//5, 0, img.shape[0],img.shape[1]*3//5],
        [0, img.shape[1]*2//5, img.shape[0]*3//5, img.shape[1]],
        [img.shape[0]*2//5, img.shape[1]*2//5, img.shape[0], img.shape[1]],
    ]
    for crop_box in crop_boxes:
        detect_result = ocr.ocr(img[crop_box[0]:crop_box[2]][crop_box[1]:crop_box[3]], rec=False, cls=True)[0]
        for bbox in detect_result:
            for point in bbox:
                point[0] += crop_box[0]
                point[1] += crop_box[1]
        result += detect_result
    # result.append(ocr.ocr(img[:img.shape[0]*3//5][:img.shape[1]*3//5], rec=False, cls=True)[0])
    # result.append(ocr.ocr(img[:img.shape[0]*3//5][img.shape[1]*2//5:], rec=False, cls=True)[0])
    # result.append(ocr.ocr(img[img.shape[0]*2//5:][:img.shape[1]*3//5], rec=False, cls=True)[0])
    # result.append(ocr.ocr(img[img.shape[0]*2//5:][img.shape[1]*2//5:], rec=False, cls=True)[0])
    
    result += ocr.ocr(img, rec=False, cls=True)[0]
    # print(result)
    clusters = make_cluster_ppocr(result, img.shape[0]//50)
    org_img = Image.open(image_path)
    mocr = MangaOcr()
    for i, cluster in enumerate(clusters):
        # print((lambda x : [x["x_min"], x["y_min"], x["x_max"], x["y_max"]])(cluster["pos"]))
        if cluster["pos"]["x_min"] == cluster["pos"]["x_max"] or cluster["pos"]["y_min"] == cluster["pos"]["y_max"]:
            continue
        crop = org_img.crop((lambda x : [x["x_min"]-10, x["y_min"]-10, x["x_max"]+10, x["y_max"]+10])(cluster["pos"]))
        
        crop.save(f"crop_{i}.jpg")
        cluster["script"] = mocr(crop)
        # print(cluster["script"] )
    return clusters
    # # draw result
    # from PIL import Image
    # result = result[0]
    # image = Image.open(image_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores)
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')

def doGoogleOCR(image_path, gc_auth):
    ocr_result = gcloud_ocr.detect_text(image_path, gc_auth)
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

def make_cluster_ppocr(det_bbox, cluster_dist_threshold=30):
    # 각 텍스트 박스의 위치관계를 이용해 클러스터링 진행
    # 현재는 박스의 4좌표를 각 각 클러스터링 하고있음
    # TODO - 이를 중간에 좌표를 임의로 만들어주거나, 중간값을 취하거나 하는 방식으로 개선 가능할 것으로 보임
    # print(det_bbox)
    bbox_points = []
    for point in det_bbox:
        # print("point", point)
        x_max = int(max([point[0][0], point[1][0], point[2][0], point[3][0]]))
        x_min = int(min([point[0][0], point[1][0], point[2][0], point[3][0]]))
        y_max = int(max([point[0][1], point[1][1], point[2][1], point[3][1]]))
        y_min = int(min([point[0][1], point[1][1], point[2][1], point[3][1]]))
        bbox_points += [[i, j] for i in range(x_min, x_max+1, 5) for j in range(y_min, y_max+1, 5)]
        # print(bbox_points)
    db_scan = DBSCAN(eps=cluster_dist_threshold, min_samples=3).fit(bbox_points)
    # print(db_scan.labels_)
    script_cluster = [{
            "script" : "", # 원본 스크립트
            "trans_script" : "", # 번역 스트립트
            "pos" : {"x_min" : 0, "x_max" : 0, "y_min" : 0, "y_max" : 0}, # 텍스트 박스 
            "raw" : [] # OCR RAW 텍스트 박스들
        } 
        for _ in range(max(db_scan.labels_) + 1)
    ]
    
    
    # OCR 된 각 텍스트 블럭을 클러스터별로 분류  
    for i, point in enumerate(bbox_points):
        label = db_scan.labels_[i]
        script_cluster[label]["raw"].append(point)
    # print(script_cluster)
    for i, script in enumerate(script_cluster):
        # print("script: ", script)
        xpos_list = set()
        ypos_list = set()
        if not script["raw"]:
            script_cluster.remove(script)
            continue
        for raw_bbox in script["raw"]:
            xpos_list.add(raw_bbox[0])
            ypos_list.add(raw_bbox[1])
        script["pos"]["x_min"] = min(xpos_list)
        script["pos"]["x_max"] = max(xpos_list)
        script["pos"]["y_min"] = min(ypos_list)
        script["pos"]["y_max"] = max(ypos_list)
        # print(script["pos"])
    # print("result: ", script_cluster)
    return script_cluster
    

def makeScript(word_data):
    # 클러스터링을 위해 좌표 노드(?) 만들기 
    text_pos = []
    for word in word_data:
        text_pos += [[word["pos"]["y_min"], word["pos"]["x_min"]],
                     [word["pos"]["y_min"], word["pos"]["x_max"]],
                     [word["pos"]["y_max"], word["pos"]["x_min"]],
                     [word["pos"]["y_max"], word["pos"]["x_max"]]]
    print(text_pos)
    # 각 텍스트 박스의 위치관계를 이용해 클러스터링 진행
    # 현재는 박스의 4좌표를 각 각 클러스터링 하고있음
    # TODO - 이를 중간에 좌표를 임의로 만들어주거나, 중간값을 취하거나 하는 방식으로 개선 가능할 것으로 보임
    db_scan = DBSCAN(eps=100, min_samples=3).fit(text_pos)

    script_cluster = [{
            "script" : "", # 원본 스크립트
            "trans_script" : "", # 번역 스트립트
            "pos" : {"x_min" : 0, "x_max" : 0, "y_min" : 0, "y_max" : 0}, # 텍스트 박스 
            "raw" : [] # OCR RAW 텍스트 박스들
        } 
        for _ in range(max(db_scan.labels_) + 2)
    ]

    # OCR 된 각 텍스트 블럭을 클러스터별로 분류  
    for i, text_block in enumerate(word_data):
        label = db_scan.labels_[i*4+2] + 1
        script_cluster[label]["raw"].append(text_block)

    for script in script_cluster:
        # 클러스터 안에 텍스트 박스가 없는 경우
        if not script["raw"]:
            continue

        # 클러스터별로 우상 -> 좌하로 정렬하여 말풍선 스크립트 생성
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

def translateScript(script_cluster, service = "google"):
    """스크립트를 번역합니다

    :param script_cluster: 클러스터링 된 스트립트
    :param service: 번역할 서비스를 선택합니다. papago, google
    :return: 번역된 스크립트
    """
    
    # if service == "papago":
    #     translator = ts.papago
    # elif service == "google":
    #     translator = ts.google
    # else:
    #     translator = ts.google
    #     script_cluster["translator"] = "google"
    # 각 스크립트별로 번역 수행
    for script in script_cluster:
        print(script["script"])
        script["trans_script"] = ts.translate_text(script["script"], translator='papago', from_language="ja", to_language="ko")
        # script["trans_script"] = translator(script["script"], from_language="ja", to_language="ko")
        # script["trans_script"] = ts.deepl(script["script"], from_language="ja", to_language="ko")
    
    return script_cluster
     
if __name__ == "__main__":
    # print(doMangaOCR("sample/blackjack_04.jpg"))
    print(do_paddlepaddle("sample/blackjack_08.jpg"))

