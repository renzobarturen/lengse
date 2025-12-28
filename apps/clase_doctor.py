import cv2

# haar cascade

# Initialize video capture
vid_capture = cv2.VideoCapture(0)

if not vid_capture.isOpened():
    print("Error opening video stream or file")
else:
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Frames per second :", fps, " Frame count :", frame_count)

while vid_capture.isOpened():
    ret, frame = vid_capture.read()

    if not ret:
        print("Video camera is disconnected")
        break

    cv2.imshow("Frame", frame)

    # Crop 256x256 centered
    crop_size = 256
    x = (frame.shape[1] - crop_size) // 2
    y = (frame.shape[0] - crop_size) // 2
    crop_roi = frame[y:y + crop_size, x:x + crop_size].copy()

    # Blur effect by resizing down and then up
    frame_downscale = cv2.resize(frame, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_CUBIC)
    frame_upscale = cv2.resize(frame_downscale, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Extract blurred crop and swap with clean crop
    blurred_crop = frame_upscale[y:y + crop_size, x:x + crop_size].copy()
    frame_upscale[y:y + crop_size, x:x + crop_size] = crop_roi

    frame_copy_with_blurred_face = frame.copy()
    frame_copy_with_blurred_face[y:y + crop_size, x:x + crop_size] = blurred_crop

    # Display
    cv2.imshow("Frame upscale", frame_upscale)
    cv2.imshow("Frame cropped", crop_roi)
    cv2.imshow("Frame cropped borrada", blurred_crop)
    cv2.imshow("Frame cropped rostro borrado", frame_copy_with_blurred_face)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        print("q key is pressed by the user. Stopping the video")
        break