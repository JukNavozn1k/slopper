"""
Standalone quality test for GENERAL mode (blurred background reframe).
No AI dependencies — just OpenCV + FFmpeg.

Usage:
    python test_general_quality.py input.mp4 output.mp4
    python test_general_quality.py input.mp4 output.mp4 --blur 51 --darken 0.5
    python test_general_quality.py input.mp4 output.mp4 --crf 18 --preset slow
    python test_general_quality.py input.mp4 output.mp4 --no-hd   # legacy 608x1080 mode
"""

import cv2
import subprocess
import os
import argparse
import numpy as np
from tqdm import tqdm

ASPECT_RATIO = 9 / 16  # width / height for 9:16 portrait


def create_general_frame(frame, output_width, output_height, blur_kernel=51, darken=0.5):
    orig_h, orig_w = frame.shape[:2]

    # --- Background: fill height, crop center, blur, darken ---
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    interp = cv2.INTER_AREA if bg_scale < 1 else cv2.INTER_LINEAR
    bg_resized = cv2.resize(frame, (bg_w, output_height), interpolation=interp)

    start_x = (bg_w - output_width) // 2
    if start_x < 0:
        start_x = 0
    background = bg_resized[:, start_x:start_x + output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height))

    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    background = cv2.GaussianBlur(background, (k, k), 0)

    if darken < 1.0:
        background = (background * darken).astype(np.uint8)

    # --- Foreground: fit width, center vertically ---
    fg_scale = output_width / orig_w
    fg_h = int(orig_h * fg_scale)
    fg_interp = cv2.INTER_AREA if fg_scale < 1 else cv2.INTER_LANCZOS4
    foreground = cv2.resize(frame, (output_width, fg_h), interpolation=fg_interp)

    y_offset = (output_height - fg_h) // 2
    final_frame = background.copy()
    final_frame[y_offset:y_offset + fg_h, :] = foreground

    return final_frame


def reframe_general(
    input_path: str,
    output_path: str,
    crf: int = 18,
    preset: str = 'fast',
    blur_kernel: int = 51,
    darken: float = 0.5,
    hd: bool = True,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if hd:
        # Standard vertical video: 1080x1920 — no platform upscaling
        out_w = 1080
        out_h = int(out_w / ASPECT_RATIO)  # 1920
    else:
        # Legacy: match input height (results in ~608x1080 for 1080p source)
        out_h = orig_h
        out_w = int(out_h * ASPECT_RATIO)
        if out_w % 2 != 0:
            out_w += 1

    print(f"Input:   {orig_w}x{orig_h} @ {fps:.2f}fps")
    print(f"Output:  {out_w}x{out_h}  CRF={crf}  preset={preset}  blur={blur_kernel}  darken={darken}")

    base = os.path.splitext(output_path)[0]
    temp_video = f"{base}_temp_video.mp4"
    temp_audio = f"{base}_temp_audio.aac"

    encode_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
        '-an', temp_video,
    ]
    ffmpeg = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    with tqdm(total=total_frames, desc="Processing") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out_frame = create_general_frame(frame, out_w, out_h, blur_kernel, darken)
            ffmpeg.stdin.write(out_frame.tobytes())
            pbar.update(1)

    ffmpeg.stdin.close()
    stderr = ffmpeg.stderr.read().decode()
    ffmpeg.wait()
    cap.release()

    if ffmpeg.returncode != 0:
        print("FFmpeg encode failed:\n", stderr)
        return False

    # Extract audio (stream copy, no re-encode)
    audio_cmd = ['ffmpeg', '-y', '-i', input_path, '-vn', '-acodec', 'copy', temp_audio]
    audio_ok = subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).returncode == 0

    if audio_ok and os.path.exists(temp_audio):
        merge_cmd = ['ffmpeg', '-y', '-i', temp_video, '-i', temp_audio,
                     '-c:v', 'copy', '-c:a', 'copy', output_path]
    else:
        merge_cmd = ['ffmpeg', '-y', '-i', temp_video, '-c:v', 'copy', output_path]

    result = subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for f in [temp_video, temp_audio]:
        if os.path.exists(f):
            os.remove(f)

    if result.returncode != 0:
        print("Merge failed:\n", result.stderr.decode())
        return False

    print(f"Done → {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test GENERAL mode reframe quality (no AI)')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--crf', type=int, default=18,
                        help='H.264 CRF: lower = better quality (default 18)')
    parser.add_argument('--preset', default='fast',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster',
                                 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='x264 preset — trades speed for compression efficiency (default fast)')
    parser.add_argument('--blur', type=int, default=51,
                        help='Gaussian blur kernel size, odd number (default 51)')
    parser.add_argument('--darken', type=float, default=0.5,
                        help='Background brightness multiplier 0.0–1.0 (default 0.5)')
    parser.add_argument('--no-hd', dest='hd', action='store_false',
                        help='Disable 1080x1920 output; use legacy input-height mode instead')
    args = parser.parse_args()

    reframe_general(
        args.input, args.output,
        crf=args.crf,
        preset=args.preset,
        blur_kernel=args.blur,
        darken=args.darken,
        hd=args.hd,
    )
