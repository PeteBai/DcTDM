import cv2
import os
import threading

def save_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Initialize frame counter
    frame_count = 0
    

    # Calculate the frame rate of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval based on the desired frame rate
    frame_interval = int(round(fps / 10))

    # Initialize frame counter
    frame_count = 0
    img_seq_ct = 0

    while True:
        # Read the next frame
        ret, frame = video.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Check if the current frame should be saved
        if frame_count % frame_interval == 0:
            # Generate the file name with a six-digit code
            file_name = f"{img_seq_ct:06d}.png"

            # Save the frame as a PNG file
            cv2.imwrite(os.path.join(output_folder, file_name), frame)
            img_seq_ct += 1
        # Increment the frame counter
        frame_count += 1

    # Release the video file
    video.release()


# List all the files in the source directory
video_files = os.listdir("/work/zura-storage/Data/DSDDM_XL/source")
# Iterate over the video files
threads = []
i = 0
for video_file in video_files:
    # Construct the full path of the video file
    video_path = os.path.join("/work/zura-storage/Data/DSDDM_XL/source", video_file)
    # Define the output folder for the frames
    output_folder = f"/work/zura-storage/Data/DSDDM_XL/images/B{i}/"
    # Call the save_frames function to extract frames from the video
    # save_frames(video_path, output_folder)
    threads.append(threading.Thread(target=save_frames, args=(video_path, output_folder)))
    threads[-1].start()
    i += 1

for th in threads:
    th.join()
print("All frames extracted successfully.")