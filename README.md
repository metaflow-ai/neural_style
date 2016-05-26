# DeepBack

# FFMPEG
Resize: `ffmpeg -i test_video.MOV -filter:v "crop=600:600:70:400" out.mp4`
video to frames: `ffmpeg -i out.mp4 -r 30/1 $filename%03d.jpeg`
frames to video: `ffmpeg -framerate 30/1 -i %03d.jpeg -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4`
