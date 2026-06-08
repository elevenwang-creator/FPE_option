"""Regenerate PDF evolution animations as APNG (full color) with a hold at the final frame."""
import subprocess, os

def regenerate(mp4_path, out_path, fps=8, hold_sec=8.0, scale=800):
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", (
            f"fps={fps},scale={scale}:-1:flags=lanczos,"
            f"tpad=stop_mode=clone:stop_duration={hold_sec}"
        ),
        "-pix_fmt", "rgba",
        "-compression_level", "1",
        out_path,
    ], check=True, capture_output=True)

    sz = os.path.getsize(out_path) / 1024
    print(f"  -> {out_path}  ({fps} fps, {hold_sec}s hold, {sz:.0f}KB, APNG)")

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "python", "examples")
    for name in ["pdf_evolution", "pdf_evolution_barrier"]:
        mp4 = os.path.join(base, f"{name}.mp4")
        out = os.path.join(base, f"{name}.apng")
        if os.path.exists(mp4):
            print(f"Regenerating {name}...")
            # Remove old .gif since we're switching to .apng
            old_gif = os.path.join(base, f"{name}.gif")
            if os.path.exists(old_gif):
                os.remove(old_gif)
            regenerate(mp4, out)
        else:
            print(f"Skipping {name}: {mp4} not found")
