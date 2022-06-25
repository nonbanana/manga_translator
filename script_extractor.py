import gcloud_ocr
from sklearn.cluster import DBSCAN
import translators as ts


# TODO - manga_ocr 테스트 필요
from manga_ocr import MangaOcr
def doMangaOCR(image_path):
    mocr = MangaOcr()
    ocr_result = mocr(image_path)
    return ocr_result

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

def makeScript(word_data):
    # 클러스터링을 위해 좌표 노드(?) 만들기 
    text_pos = []
    for word in word_data:
        text_pos += [[word["pos"]["y_min"], word["pos"]["x_min"]],
                     [word["pos"]["y_min"], word["pos"]["x_max"]],
                     [word["pos"]["y_max"], word["pos"]["x_min"]],
                     [word["pos"]["y_max"], word["pos"]["x_max"]]]
    
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
    
    if service == "papago":
        translator = ts.papago
    elif service == "google":
        translator = ts.google
    else:
        translator = ts.google
        script_cluster["translator"] = "google"

    # 각 스크립트별로 번역 수행
    for script in script_cluster:
        script["trans_script"] = translator(script["script"], from_language="ja", to_language="ko")
        # script["trans_script"] = ts.deepl(script["script"], from_language="ja", to_language="ko")
    
    return script_cluster
        

