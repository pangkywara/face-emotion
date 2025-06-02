import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
print(f"Loaded NumPy version: {np.__version__}")
from collections import OrderedDict
import sys
import os
import tensorflow as tf # Added for RetinaFace
from itertools import product # Added for RetinaFace post-processing
from math import ceil # Added for RetinaFace post-processing

print("--- sys.path --- ")
print(sys.path)
print("----------------")

# Import the custom class needed for loading the checkpoint
from main_QCS_raf_db import RecorderMeter_matrix, RecorderMeter_loss

# Attempt to import scipy to check its path *before* sklearn does
try:
    import scipy
    print(f"Attempting to import scipy: {scipy.__file__}")
except Exception as e:
    print(f"Could not pre-import scipy: {e}")

# Ensure the QCS directory is in the Python path
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the workspace root (assuming script is in the root)
workspace_root = script_dir
# Construct the absolute path to the QCS directory
qcs_dir = os.path.join(workspace_root, 'QCS')
# Add the workspace root to sys.path if QCS can be imported directly
if qcs_dir not in sys.path:
    sys.path.insert(0, workspace_root) # Add root for 'from QCS...' imports

# Now import the model definition
try:
    from QCS.QCS_7cls_raf_db import pyramid_trans_expr
except ImportError as e:
    print(f"Error importing model definition: {e}")
    print("Please ensure:")
    print("1. You are running this script from the 'face-emote' directory (workspace root).")
    print(f"2. The QCS directory exists at: {qcs_dir}")
    print("3. __init__.py files exist in 'QCS' and 'QCS/models'.")
    print(f"4. Python can find the QCS module (check sys.path: {sys.path})")
    exit()


# --- Configuration ---
EMOTION_MODEL_PATH = 'QCS/me_models/[12-16]-[09-38]-model_best.pth'
EMOTION_LABELS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'] # RAF-DB 7 classes
NUM_CLASSES = 7
INPUT_SIZE_EMOTION = 224 # For emotion model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for PyTorch models: {DEVICE}")

# Face Detection Model (RetinaFace)
RETINAFACE_SAVED_MODEL_PATH = 'dodosd/model/retinaface_mobilefacenet025_tf'
RETINAFACE_INPUT_SIZE = 480 # Expected input size for RetinaFace
CONF_THRESHOLD_FACE = 0.5 # Confidence threshold for face detection
NMS_THRESHOLD_FACE = 0.4  # NMS threshold for face detection

# --- RetinaFace Helper Configuration & Functions (Adapted from dodosd/realtime_detect.py) ---
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
                    # dense_cx = [x * self.steps[k] / self.image_size_w for x in [j + 0.5]] # Original, check if image_size[1] was width
                    # dense_cy = [y * self.steps[k] / self.image_size_h for y in [i + 0.5]] # Original, check if image_size[0] was height
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


# --- Check if files exist ---
if not os.path.exists(EMOTION_MODEL_PATH):
    print(f"Error: Emotion model file not found at {EMOTION_MODEL_PATH}")
    exit()
# Face detection model (directory check)
if not os.path.isdir(RETINAFACE_SAVED_MODEL_PATH):
    print(f"Error: RetinaFace model directory not found at {RETINAFACE_SAVED_MODEL_PATH}")
    exit()


# --- Load RetinaFace Detector (TensorFlow SavedModel) ---
try:
    print(f"Loading RetinaFace model from {RETINAFACE_SAVED_MODEL_PATH}...")
    retinaface_model_loaded = tf.saved_model.load(RETINAFACE_SAVED_MODEL_PATH)
    retinaface_infer = retinaface_model_loaded.signatures['serving_default']
    
    # Try to get input tensor name
    try:
        # Preferred way if available
        retinaface_input_name = list(retinaface_infer.structured_input_signatures[0].keys())[0]
    except AttributeError:
        # Fallback: Try to get it from the concrete function's inputs if the above fails
        # This assumes the first input to the concrete function is our target
        if hasattr(retinaface_infer, 'inputs') and retinaface_infer.inputs:
            retinaface_input_name = retinaface_infer.inputs[0].name.split(':')[0] # Get name before colon
        else:
            # Last resort: If we can't find it, we might have to hardcode or use a common default
            # For models converted from ONNX, 'input_1' or the original ONNX input name is common
            print("Warning: Could not dynamically determine RetinaFace input name. Assuming a common name like 'input_1' or first arg.")
            # We will try to call it and see if TF can bind it. This might still fail.
            # If the serving_default is a concrete function, its first argument name might be used by TF implicitly.
            # For now, let the call be retinaface_infer(**{SOME_ASSUMED_NAME: input_tensor_retina})
            # or just try positional if the kwargs method fails later.
            # To make the ** call work, we need *some* name. Let's try a common placeholder.
            retinaface_input_name = "input_1" # Placeholder, may need adjustment

    print(f"RetinaFace model loaded. Attempting to use input tensor name: {retinaface_input_name}")
except Exception as e:
    print(f"Error loading RetinaFace TensorFlow SavedModel: {e}")
    exit()


# --- Define Image Transformations (For Emotion Model) ---
emotion_data_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE_EMOTION, INPUT_SIZE_EMOTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Emotion image transformations defined.")

# --- Load Emotion Recognition Model ---
try:
    print("Loading emotion model definition...")
    emotion_model = pyramid_trans_expr(img_size=INPUT_SIZE_EMOTION, num_classes=NUM_CLASSES)
    print("Emotion model definition loaded.")

    print(f"Loading emotion model weights from {EMOTION_MODEL_PATH}...")
    checkpoint = torch.load(EMOTION_MODEL_PATH, map_location=DEVICE)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Extracted 'state_dict' from emotion checkpoint.")
    elif isinstance(checkpoint, OrderedDict):
         state_dict = checkpoint
         print("Using emotion checkpoint directly as state_dict.")
    else:
        raise KeyError("Could not find 'state_dict' or suitable state dictionary in the emotion checkpoint.")

    new_state_dict = OrderedDict()
    keys_cleaned = 0
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
            new_state_dict[name] = v
            keys_cleaned += 1
        else:
            new_state_dict[k] = v
    if keys_cleaned > 0:
        print(f"Removed 'module.' prefix from {keys_cleaned} keys in emotion model.")

    emotion_model.load_state_dict(new_state_dict, strict=False)
    print("Emotion model weights loaded successfully (non-strict mode).")

    emotion_model.to(DEVICE)
    emotion_model.eval()
    print("Emotion model moved to device and set to evaluation mode.")

except FileNotFoundError:
    print(f"Error: Emotion model file not found at {EMOTION_MODEL_PATH}")
    exit()
except KeyError as e:
    print(f"Error loading emotion model state_dict: Missing key {e}.")
    exit()
except Exception as e:
    print(f"An error occurred loading the emotion model: {e}")
    import traceback
    traceback.print_exc()
    exit()


# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam initialized. Starting real-time detection...")
print("Press 'q' to quit.")

# --- Create PriorBox for RetinaFace ---
# The image size for priorbox should be (width, height) matching RetinaFace input
priorbox = PriorBoxNumpy(cfg_mnet, image_size_wh=(RETINAFACE_INPUT_SIZE, RETINAFACE_INPUT_SIZE))
priors = priorbox.forward()


# --- Real-time Detection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_height, frame_width, _ = frame.shape
    
    # --- Face Detection with RetinaFace ---
    # 1. Preprocess frame for RetinaFace
    img_for_retina = cv2.resize(frame, (RETINAFACE_INPUT_SIZE, RETINAFACE_INPUT_SIZE))
    img_for_retina_rgb = cv2.cvtColor(img_for_retina, cv2.COLOR_BGR2RGB) # Model might expect RGB
    img_for_retina_tensor = img_for_retina_rgb.astype(np.float32)
    # Mean subtraction as per common RetinaFace setups (adjust if different)
    # Using values from dodosd/realtime_detect.py (implicitly, by not re-adding them to image read by PIL)
    # For TensorFlow, it's common to normalize to [-1, 1] or [0, 1] or use specific means.
    # The dodosd ONNX conversion was from PyTorch which often uses 0-255 and then specific normalization.
    # Let's assume for now the TF SavedModel takes BGR 0-255 and handles internal conversion,
    # or expects RGB 0-255. Common TF practice is often specific channel order and normalization.
    # From dodosd README, PyTorch input was NCHW. TF is usually NHWC.
    # Let's use the preprocessing from dodosd/realtime_detect.py: BGR, float32, mean subtraction [104, 117, 123]
    
    # Re-check dodosd/realtime_detect.py for exact preprocessing for its TFLite model
    # It loads with PIL (RGB), converts to np.float32, then (img - 127.5) / 128.0 for [-1,1]
    # OR, it uses the original PyTorch means.
    # The ONNX conversion from PyTorch means the ONNX model expects PyTorch-style input.
    # The TF SavedModel from ONNX might retain this or change it.
    # For now, let's try BGR with mean subtraction.
    prep_img = img_for_retina.astype(np.float32)
    prep_img -= (104, 117, 123) # BGR mean subtraction
    prep_img = np.transpose(prep_img, (2, 0, 1)) # Transpose from HWC to CHW
    
    input_tensor_retina = tf.convert_to_tensor(np.expand_dims(prep_img, axis=0)) # Now NHWC means NCHW

    # 2. Perform RetinaFace Inference
    # The output of SavedModel might be a dict. We need to know keys.
    # Common output names from TF conversion of RetinaFace ONNX: 'boxes', 'scores', 'landmarks'
    # Or it might be a list of tensors in order: loc, conf, landms
    try:
        # Check model's callable signatures if direct call fails
        # print(retinaface_model.signatures) -> run this once to see available signatures
        # For now, assume direct call or 'serving_default' is used implicitly by direct call
        # And assume output is a list [loc, conf, landms] or [loc, landms, conf] etc.
        # The dodosd/realtime_detect.py (TFLite) gets three outputs.
        # Output order for RetinaFace from Pytorch_Retinaface: loc, conf, landms
        
        # Call the signature with the input tensor as a keyword argument
        raw_detections = retinaface_infer(**{retinaface_input_name: input_tensor_retina})

        if isinstance(raw_detections, dict) and all(k in raw_detections for k in ['loc', 'conf', 'landms']):
            loc_data = raw_detections['loc'].numpy()
            conf_data = raw_detections['conf'].numpy()
            landms_data = raw_detections['landms'].numpy()
        else:
            print("Error: RetinaFace output dictionary did not contain expected keys ('loc', 'conf', 'landms').")
            loc_data = np.zeros((1,100,4)) # Dummy
            conf_data = np.zeros((1,100,2)) # Dummy
            landms_data = np.zeros((1,100,10)) # Dummy

        scale = np.array([frame_width, frame_height, frame_width, frame_height])
        scale_landms = np.array([
            frame_width, frame_height, frame_width, frame_height,
            frame_width, frame_height, frame_width, frame_height,
            frame_width, frame_height
        ])

        boxes = decode_boxes_numpy(loc_data[0], priors, cfg_mnet['variance'])
        boxes = boxes * scale # Scale to original frame size
        
        scores = conf_data[0][:, 1] # Confidence for 'face' class (index 1)

        landms = decode_landm_numpy(landms_data[0], priors, cfg_mnet['variance'])
        landms = landms * scale_landms

        # Ignore low scores
        valid_idx = np.where(scores > CONF_THRESHOLD_FACE)[0]
        boxes = boxes[valid_idx]
        scores = scores[valid_idx]
        landms = landms[valid_idx]

        # Keep top-k before NMS (optional, helps performance if many detections)
        # order = scores.argsort()[::-1][:5000] # top 5k
        # boxes = boxes[order]
        # scores = scores[order]
        # landms = landms[order]

        # Do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep_indices = py_cpu_nms(dets, NMS_THRESHOLD_FACE)
        final_faces = dets[keep_indices, :]
        # final_landms = landms[keep_indices, :] # If you need landmarks

        detected_face_coords = []
        for b in final_faces:
            # b is [x1, y1, x2, y2, score]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            w, h = x2 - x1, y2 - y1
            # Ensure valid box dimensions
            if w > 0 and h > 0:
                 detected_face_coords.append((x1, y1, w, h))
        
    except Exception as e:
        print(f"Error during RetinaFace detection: {e}")
        import traceback
        traceback.print_exc()
        detected_face_coords = [] # Fallback to no faces if error


    # --- Process each detected face (using coords from RetinaFace) ---
    for (x, y, w, h) in detected_face_coords:
        # Ensure coordinates are within frame boundaries
        x, y = max(0, x), max(0, y)
        w, h = min(frame_width - x, w), min(frame_height - y, h)
        if w <=0 or h <=0: continue

        face_roi_bgr = frame[y:y+h, x:x+w]
        if face_roi_bgr.size == 0: continue # Skip if ROI is empty

        face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_roi_rgb)
        input_tensor_emotion = emotion_data_transform(pil_img)
        input_batch_emotion = input_tensor_emotion.unsqueeze(0).to(DEVICE)

        emotion_label = "Unknown"
        try:
            with torch.no_grad():
                logits = emotion_model(input_batch_emotion)
                probabilities = torch.softmax(logits, dim=1)
                # if DEVICE.type == 'cuda': # Print statement can be removed now if not needed
                #     print("Emotion probabilities:", probabilities.cpu().numpy().flatten())
                # else:
                #     print("Emotion probabilities:", probabilities.numpy().flatten())
                
                confidence, pred_idx = torch.max(probabilities, 1)
                original_pred_idx = pred_idx.item()
                original_confidence = confidence.item()

                # Map 7 classes to 4 classes
                # Original: ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'] (0-6)
                # Target:   ['Happy', 'Sad', 'Anger', 'Neutral']
                if original_pred_idx == 3: # Happiness
                    mapped_label = 'Happy'
                elif original_pred_idx == 1 or original_pred_idx == 4: # Fear or Sadness
                    mapped_label = 'Sad'
                elif original_pred_idx == 5: # Anger
                    mapped_label = 'Anger'
                elif original_pred_idx == 0 or original_pred_idx == 2 or original_pred_idx == 6: # Surprise, Disgust, Neutral
                    mapped_label = 'Neutral'
                else: # Should not happen with 7 classes
                    mapped_label = 'Unknown' 

                # Use mapped label and original confidence for display
                emotion_label = f"{mapped_label} ({original_confidence*100:.1f}%)"
                # print(f"Predicted Emotion: {emotion_label}") # Print statement can be removed now if not needed

        except Exception as e:
            print(f"Error during emotion inference: {e}")
            emotion_label = "Error"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Real-time Emotion Detection (Press Q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Exiting...")
cap.release()
cv2.destroyAllWindows()
print("Resources released.") 