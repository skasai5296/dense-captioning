import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, help="Path to videos")
    parser.add_argument("-s", "--json_path", type=str, help="Path to json file")
    parser.add_argument("-o", "--save_path", type=str, help="Path to json file to save")
    args = parser.parse_args()
    modes = ["train", "val_1", "val_2"]

    video_dir_path = Path(args.video_path)
    json_path = Path(args.json_path)
    if args.save_path:
        dst_json_path = Path(args.save_path)
    else:
        dst_json_path = json_path

    with json_path.open("r") as f:
        json_data = json.load(f)

    for i, video_file_path in enumerate(sorted(video_dir_path.iterdir())):
        file_name = video_file_path.name
        if ".mp4" not in file_name:
            continue
        id = video_file_path.stem

        ffprobe_cmd = ["ffprobe", str(video_file_path)]
        p = subprocess.Popen(
            ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        res = p.communicate()[1].decode("utf-8")

        fps = float([x for x in res.split(",") if "fps" in x][0].rstrip("fps"))
        if id in json_data.keys():
            json_data[id]["fps"] = fps
        if i % 100 == 99:
            print("{}/{}".format(i + 1, len(os.listdir(video_dir_path))))

    with dst_json_path.open("w") as f:
        json.dump(json_data, f)
