import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import cv2
import time
import argparse
import os
import json

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def main(result, video):
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(video)
        else:
            cap = cv2.VideoCapture(video)
        cap.set(3, 512)
        cap.set(4, 512)

        start = time.time()
        frame_count = 0
        k = 0
    
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            keypoints = []
            for i in keypoint_coords[0]:
                for j in i :
                    keypoints.append(j)
            # print(keypoints)
            keypoint_dict = {'keypoint' : keypoints}

            if not os.path.isdir(result + video.split('/')[-1].split('.')[0]):                                                           
                os.mkdir(result + video.split('/')[-1].split('.')[0])
            with open(result + video.split('/')[-1].split('.')[0] + '/' + video.split('/')[-1].split('.')[0] + '_' + str(k) +'.json', "w") as json_file:
                json.dump(keypoint_dict, json_file)
            k += 1

            frame_count += 1
            if frame_count == 150:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(result_folder, input_path)
