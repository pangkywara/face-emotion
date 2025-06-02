import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import os
import tensorflow as tf # Added for RetinaFace
from itertools import product # Added for RetinaFace post-processing
from math import ceil # Added for RetinaFace post-processing

# --- Configuration & Constants ---
# Model paths (adjust if needed)
DEFAULT_ONNX_MODEL_PATH = 'QCS/me_models/mobilenetv3_mixed_weighted_fp16.onnx' # Use the improved weighted FP16 model
DEFAULT_CASCADE_PATH = 'QCS/me_models/haarcascade_frontalface_default.xml'

INPUT_SIZE = 224 # Model input size
EMOTION_LABELS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# Normalization parameters (same as used in training/evaluation)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Face Detection Model (RetinaFace TensorFlow)
RETINAFACE_SAVED_MODEL_PATH = 'dodosd/model/retinaface_mobilefacenet025_tf'
RETINAFACE_INPUT_SIZE = 480 # Expected input size for RetinaFace
CONF_THRESHOLD_FACE = 0.5 # Confidence threshold for face detection
NMS_THRESHOLD_FACE = 0.4  # NMS threshold for face detection

# --- RetinaFace Helper Configuration & Functions ---
# (Adapted from dodosd/realtime_detect.py / realtime_test.py modifications)
cfg_mnet = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2], # Variance for box decoding
    'clip': False, # Clip box coordinates to [0, 1]?
    'image_size': RETINAFACE_INPUT_SIZE
}

class PriorBoxNumpy(object):
    def __init__(self, cfg, image_size_wh): # image_size_wh should be (width, height)
        super(PriorBoxNumpy, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size_w, self.image_size_h = image_size_wh[0], image_size_wh[1]
        self.feature_maps = [[ceil(self.image_size_h/step), ceil(self.image_size_w/step)] for step in self.steps] # h, w

    def forward(self):
        anchors = []
        for k, f_hw in enumerate(self.feature_maps): # f_hw is [feature_map_h, feature_map_w]
            f_h, f_w = f_hw[0], f_hw[1]
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f_h), range(f_w)): # i for height, j for width
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size_w
                    s_ky = min_size / self.image_size_h
                    dense_cx = [(j + 0.5) * self.steps[k] / self.image_size_w]
                    dense_cy = [(i + 0.5) * self.steps[k] / self.image_size_h]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        return output

def decode_boxes_numpy(loc, priors, variances):
    boxes = np.concatenate((\
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],\
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm_numpy(pre, priors, variances):
    landms = np.concatenate((\
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],\
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],\
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],\
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],\
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],\
        ), axis=1)
    return landms

def py_cpu_nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
# --- End RetinaFace Helpers ---

def preprocess_face(face_img, target_np_type=np.float16):
    """Preprocesses the cropped face image for the ONNX model."""
    # Resize
    resized_face = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    # Convert BGR to RGB
    rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    # Convert to float32 and scale to [0, 1]
    float_face = rgb_face.astype(np.float32) / 255.0
    # Normalize
    normalized_face = (float_face - NORM_MEAN) / NORM_STD
    # Transpose dimensions from HWC to CHW (required by PyTorch/ONNX models)
    chw_face = normalized_face.transpose(2, 0, 1)
    # Add batch dimension (BCHW)
    batch_face = np.expand_dims(chw_face, axis=0)
    # Convert to the target numpy type (e.g., float16)
    typed_face = batch_face.astype(target_np_type)
    return typed_face

def preprocess_emotion_face(face_img, target_np_type=np.float16):
    """Preprocesses the cropped face image for the ONNX emotion model."""
    # Resize
    resized_face = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    # Convert BGR to RGB
    rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    # Convert to float32 and scale to [0, 1]
    float_face = rgb_face.astype(np.float32) / 255.0
    # Normalize
    normalized_face = (float_face - NORM_MEAN) / NORM_STD
    # Transpose dimensions from HWC to CHW (required by PyTorch/ONNX models)
    chw_face = normalized_face.transpose(2, 0, 1)
    # Add batch dimension (BCHW)
    batch_face = np.expand_dims(chw_face, axis=0)
    # Convert to the target numpy type (e.g., float16)
    typed_face = batch_face.astype(target_np_type)
    return typed_face

def main(args):
    onnx_model_path = args.onnx_model_path
    # cascade_path = args.cascade_path # Removed

    # --- Validate Paths ---
    if not os.path.exists(onnx_model_path):
        print(f"Error: ONNX emotion model not found at {onnx_model_path}")
        return
    # if not os.path.exists(cascade_path): # Removed
    #     print(f"Error: Haar Cascade file not found at {cascade_path}")
    #     return
    if not os.path.isdir(RETINAFACE_SAVED_MODEL_PATH): # Changed validation
        print(f"Error: RetinaFace model directory not found at {RETINAFACE_SAVED_MODEL_PATH}")
        return

    # --- Load ONNX Emotion Model Session ---
    try:
        print(f"Loading ONNX emotion model from {onnx_model_path}...")
        providers = ['CPUExecutionProvider'] # Default to CPU
        # if 'CUDAExecutionProvider' in ort.get_available_providers():
        #     providers.insert(0, 'CUDAExecutionProvider')

        emotion_session = ort.InferenceSession(onnx_model_path, providers=providers)
        emotion_input_name = emotion_session.get_inputs()[0].name
        emotion_output_name = emotion_session.get_outputs()[0].name
        emotion_input_type_str = emotion_session.get_inputs()[0].type
        emotion_target_np_type = np.float16 if emotion_input_type_str == 'tensor(float16)' else np.float32
        print(f"ONNX emotion session loaded. Input: '{emotion_input_name}' ({emotion_input_type_str}), Output: '{emotion_output_name}', Provider: {emotion_session.get_providers()}")
        print(f"Emotion input data will be converted to: {emotion_target_np_type}")

    except Exception as e:
        print(f"Error loading ONNX emotion model: {e}")
        return

    # --- Load RetinaFace Detector (TensorFlow SavedModel) ---
    try:
        print(f"Loading RetinaFace model from {RETINAFACE_SAVED_MODEL_PATH}...")
        retinaface_model_loaded = tf.saved_model.load(RETINAFACE_SAVED_MODEL_PATH)
        retinaface_infer = retinaface_model_loaded.signatures['serving_default']
        
        # Try to get input tensor name
        try:
            retinaface_input_name = list(retinaface_infer.structured_input_signatures[0].keys())[0]
        except AttributeError:
            if hasattr(retinaface_infer, 'inputs') and retinaface_infer.inputs:
                retinaface_input_name = retinaface_infer.inputs[0].name.split(':')[0]
            else:
                print("Warning: Could not dynamically determine RetinaFace input name. Assuming 'input_1'.")
                retinaface_input_name = "input_1"

        print(f"RetinaFace model loaded. Attempting to use input tensor name: {retinaface_input_name}")
    except Exception as e:
        print(f"Error loading RetinaFace TensorFlow SavedModel: {e}")
        exit()


    # --- Initialize Webcam ---
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam initialized.")

    # --- Create PriorBox for RetinaFace ---
    priorbox = PriorBoxNumpy(cfg_mnet, image_size_wh=(RETINAFACE_INPUT_SIZE, RETINAFACE_INPUT_SIZE))
    priors = priorbox.forward()

    # --- Real-time Loop ---
    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_height, frame_width, _ = frame.shape
        detected_face_coords = [] # Initialize list for detected faces this frame

        # --- Face Detection with RetinaFace ---
        try:
            # 1. Preprocess frame for RetinaFace (NCHW format)
            img_for_retina = cv2.resize(frame, (RETINAFACE_INPUT_SIZE, RETINAFACE_INPUT_SIZE))
            prep_img = img_for_retina.astype(np.float32)
            prep_img -= (104, 117, 123) # BGR mean subtraction
            prep_img = np.transpose(prep_img, (2, 0, 1)) # HWC -> CHW
            input_tensor_retina = tf.convert_to_tensor(np.expand_dims(prep_img, axis=0)) # NCHW

            # 2. Perform RetinaFace Inference
            raw_detections = retinaface_infer(**{retinaface_input_name: input_tensor_retina})

            # 3. Parse Output
            if isinstance(raw_detections, dict) and all(k in raw_detections for k in ['loc', 'conf', 'landms']):
                loc_data = raw_detections['loc'].numpy()
                conf_data = raw_detections['conf'].numpy()
                landms_data = raw_detections['landms'].numpy() # Not used below, but parsed
            else:
                print("Warning: Unexpected RetinaFace output format. Skipping face detection this frame.")
                loc_data, conf_data, landms_data = None, None, None # Ensure they exist for logic below

            # 4. Postprocess (if data was parsed correctly)
            if loc_data is not None:
                scale = np.array([frame_width, frame_height, frame_width, frame_height])
                # scale_landms = np.array([frame_width, frame_height] * 5) # Simpler way

                boxes = decode_boxes_numpy(loc_data[0], priors, cfg_mnet['variance'])
                boxes = boxes * scale
                scores = conf_data[0][:, 1] # Confidence for 'face' class

                # Ignore low scores
                valid_idx = np.where(scores > CONF_THRESHOLD_FACE)[0]
                if len(valid_idx) > 0:
                    boxes = boxes[valid_idx]
                    scores = scores[valid_idx]
                    # landms_data is not used currently, but could be decoded here if needed

                    # NMS
                    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep_indices = py_cpu_nms(dets, NMS_THRESHOLD_FACE)
                    final_faces = dets[keep_indices, :]

                    for b in final_faces:
                        # Clamp box coordinates to frame dimensions before converting to int
                        x1 = max(0, int(b[0]))
                        y1 = max(0, int(b[1]))
                        x2 = min(frame_width, int(b[2]))
                        y2 = min(frame_height, int(b[3]))
                        w = x2 - x1
                        h = y2 - y1
                        if w > 0 and h > 0:
                             detected_face_coords.append((x1, y1, w, h))

        except Exception as e:
            print(f"Error during RetinaFace detection: {e}")
            import traceback
            traceback.print_exc()
            # Keep detected_face_coords empty

        # --- Emotion Recognition for each detected face ---
        for (x, y, w, h) in detected_face_coords:
            # No need to re-clamp here if done above after NMS
            face_roi = frame[y:y+h, x:x+w] 

            if face_roi.size == 0: continue

            try:
                # Preprocess the face for the ONNX emotion model
                preprocessed_face_emotion = preprocess_emotion_face(face_roi, emotion_target_np_type)

                # Run ONNX emotion inference
                onnx_outputs = emotion_session.run([emotion_output_name], {emotion_input_name: preprocessed_face_emotion})[0]

                # Get prediction
                predicted_idx = np.argmax(onnx_outputs, axis=1)[0]
                predicted_emotion = EMOTION_LABELS[predicted_idx]
                # Calculate softmax confidence from logits
                exp_outputs = np.exp(onnx_outputs[0] - np.max(onnx_outputs[0])) # Subtract max for numerical stability
                softmax_probs = exp_outputs / np.sum(exp_outputs)
                confidence = np.max(softmax_probs)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_text = f'{predicted_emotion} ({confidence:.2f})'
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face for emotion: {e}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0: # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Real-time Emotion Detection (RetinaFace + ONNX)', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time Emotion Detection using RetinaFace (TF) and an ONNX emotion model.')
    parser.add_argument('--onnx_model_path', type=str, default=DEFAULT_ONNX_MODEL_PATH,
                        help=f'Path to the ONNX emotion model file (default: {DEFAULT_ONNX_MODEL_PATH})')
    # Removed cascade_path and related arguments
    # Add arguments for RetinaFace thresholds if needed, otherwise use constants
    # parser.add_argument('--conf_thresh', type=float, default=CONF_THRESHOLD_FACE, ...)
    # parser.add_argument('--nms_thresh', type=float, default=NMS_THRESHOLD_FACE, ...)

    args = parser.parse_args()
    main(args) 