import cv2
import torch
import numpy as np

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def tensor_to_bgr(frame_tensor, *, undo_normalise=False, mean=None, std=None):
    """
    Args
    ----
    frame_tensor : torch.Tensor
        (C,H,W) or (1,C,H,W)   ―  float or half   ―  RGB
    undo_normalise : bool
        True if you previously applied (x - mean) / std
    mean, std : list/tuple of 3 floats
        Same numbers you used for normalising (e.g. ImageNet)
    Returns
    -------
    frame_bgr : np.ndarray   (H,W,3) uint8   BGR  contiguous
    """
    # 1) squeeze batch dimension if present
    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]

    # 2) move to CPU & float32 for math
    img = frame_tensor.detach()

    # 3) (optional) reverse mean/std normalisation
    if undo_normalise:
        if mean is None or std is None:
            raise ValueError("Supply mean and std to undo normalisation")
        mean = torch.tensor(mean).to(img).view(3,1,1)
        std  = torch.tensor(std).to(img).view(3,1,1)
        img = img * std + mean

    # 4) scale back to 0‑255, clamp, uint8
    img = (img * 255.0)
    # img = img # .to(torch.float16)
    img = img.clamp(0,255).byte()

    # 5) channel‑last & numpy
    img = img.permute(1,2,0).cpu().numpy()                 # H,W,C  RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # → BGR
    img = np.ascontiguousarray(img)                  # ensure OpenCV‑happy
    return img


def load_model(model_path):
    model = torch.jit.load(model_path)
    model.to(torch.device('mps'))
    # model.eval()
    return model

def frame_to_tensor(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3) Ensure the array is contiguous (torch needs it) -------------------------
    frame = np.ascontiguousarray(frame_rgb)

    # 4) numpy -> torch, move channels, scale, add batch if wanted --------------
    tensor = torch.from_numpy(frame)     # H x W x C, uint8 → int tensor
    tensor = tensor.to('mps', non_blocking=True)
    
    tensor = tensor.permute(2, 0, 1)         # C x H x W
    tensor = tensor.to(torch.float32).div(255.0)       # float32, [0,1]

    # 5) (Optional) add a batch dim and push to GPU ------------------------------
    tensor = tensor.unsqueeze(0)             # 1 x C x H x W
    return tensor.to(torch.float16)

def model_inference(model, tensor):
    print(tensor.shape)
    return model(tensor) / 255.0


def main():
    # Load pre-trained Haar cascade classifier for frontal face detection
    # haarcascade_profileface
    # haarcascade_frontalface_default
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture from the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    # starry_v_bt4_1e10_ep2_float16
    # nature_oil_painting_ep3_bt4_sw3e10_cw_1e5_float16
    # starry_ep3_bt4_sw1e11_cw_1e5_float16 <- one of the better ones
    # nature_oil_painting_ep4_bt4_sw1e10_cw_1e5_float16
    model = load_model('../models/exports/mps/starry_ep3_bt4_sw1e11_cw_1e5_float16.pt')

    print("Press 'q' to quit")
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break

        # Convert the frame to grayscale (face detector expects gray images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        for n in [5,3,1,0]:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=n,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces):
                break

        x_grow = 1.6
        y_grow = 1.9
        height, width, channels = frame.shape

        face_bounds = []
        for (i, (x, y, w, h)) in enumerate(faces):
            # Calculate the center of the face
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate the new width and height
            new_w = int(w * x_grow)
            new_h = int(h * y_grow)

            # Calculate the new top-left corner
            new_x = max(0, center_x - new_w // 2)
            new_y = max(0, center_y - new_h // 2)

            # Ensure the new bounding box is within the image boundaries
            new_x = min(new_x, width - new_w)
            new_y = min(new_y, height - new_h)

            face_bounds.append((new_x, new_y, new_w, new_h))

        # Draw bounding boxes around detected faces
        for (x, y, w, h) in face_bounds:
            face_roi = frame[y:y+h, x:x+w]
            # Apply style transfer to the face region
            input_tensor = frame_to_tensor(face_roi)
            print(input_tensor.shape)
            output_tensor = model_inference(model, input_tensor)[:,:,0:h, 0:w]
            # print(output_tensor.shape)
            output_face = tensor_to_bgr(output_tensor)
            # Replace the face region in the original frame with the stylized face
            frame[y:y+h, x:x+w] = output_face # output_face[:h, :w]
            # draw_border(frame, (x, y), (x + w, y + h), (255,255,127), 2, 20, 20)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Webcam Face Detection', frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
