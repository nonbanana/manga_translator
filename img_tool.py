from PIL import ImageDraw


def drawOCRText(input_img, ocr_response, font):
    """
    draw text from ocr response
    :img_src: PIL 이미지 소스
    :ocr_response: OCR 이미지 결과
    """
    draw = ImageDraw.Draw(input_img)
    text_annotation = ocr_response.text_annotations
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

