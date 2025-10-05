import os
from exif_process.exif_reader import get_af_info
import cv2
import logging
import shutil
from blur_detection.detection import BirdSharpnessAnalyzer

time_threshold=5
folder = r"./data/origin_photo"
new_folder = r"./data/selected"
tmp_folder = r"./data/tmp"
if not os.path.exists(new_folder):
    os.mkdir(new_folder)
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

if __name__ == "__main__":
    photos=[]
    exif_info=dict()
    for file in os.listdir(folder):
        if file.endswith(".CR3"):
            photos.append(file)

    for photo in photos:
        exif_info[photo]=get_af_info(os.path.join(folder,photo))
    # 按拍摄时间排序
    exif_info_sorted=dict(sorted(exif_info.items(), key=lambda item: item[1][0]))

    # 分组处理
    groups = []
    current_group = []
    prev_time = None

    for photo in exif_info_sorted:
        if prev_time is None:
            current_group.append(photo)
        else:
            time_diff = (exif_info_sorted[photo][0] - prev_time).total_seconds()
            if time_diff <= time_threshold:
                current_group.append(photo)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [photo]
        prev_time = exif_info_sorted[photo][0]

    if current_group:
        groups.append(current_group)

    analyzer = BirdSharpnessAnalyzer(model_path='../yolov8l-seg.pt')

    for gp in groups:
        print(f'photo group: {",".join(gp)}')
        score_list=[]
        for image_path in gp:
            image_path = os.path.join(folder,image_path.split(".")[0])+'.JPG'

            image = cv2.imread(str(image_path))
            if image is None:
                logging.warning(f'warning! failed to read image from {image_path}; skipping!')
                continue

            logging.info(f'processing {image_path}')
            visualized_image, analysis_results = analyzer.analyze_image(image_path)

            if len(analysis_results)==0:
                score=0
            else:
                score=max([result['sharpness'] for result in analysis_results])

            print(f'image_path: {image_path} score: {score}')

            # output marked picture for debug
            # output_path = os.path.join(tmp_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
            # cv2.imwrite(output_path, visualized_image)
            score_list.append(score)

        if len(score_list):
            print(f'best picture: {gp[score_list.index(max(score_list))]}')
            best_file=gp[score_list.index(max(score_list))]
            shutil.copyfile(os.path.join(folder,best_file), os.path.join(new_folder,best_file))




