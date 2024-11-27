import subprocess
import glob

def get_frame_count_ffmpeg(video_file):
    # ffprobeを使って動画のフレーム数を取得するコマンド
    command = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-count_frames', 
        '-show_entries', 'stream=nb_read_frames', 
        '-of', 'default=nokey=1:noprint_wrappers=1', 
        video_file
    ]
    
    try:
        # コマンドを実行して結果を取得
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
        frame_count = int(output.strip())
        return frame_count
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

# 動画ファイルのパスパターンを指定して一致するファイルを取得
video_files = glob.glob("data/picture/**/*.mp4", recursive=True)
max_frames = 0
cnt = 0
# 各動画ファイルについてフレーム数を取得
for video_file in video_files:
    frame_count = get_frame_count_ffmpeg(video_file)
    cnt += 1
    print(cnt)
    if frame_count is not None and frame_count > max_frames:
        max_frames = frame_count

print(f"最大フレーム数: {max_frames}")