# filename: run_vsr_cli.py
# Purpose: Command-line wrapper for video-subtitle-remover's backend
#          to enable automation via subprocess calls.
#          Accepts multiple video files and an optional fixed subtitle area.
#          Must be run using the dedicated VSR Python 3.8 environment.

import argparse
import os
import sys
import time
from pathlib import Path

# --- Crucial: Ensure 'backend' can be imported ---
# Add the directory containing this script (which should be the root of
# the video-subtitle-remover project) to the Python path.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
# --- End Path Setup ---

try:
    # Attempt to import the core VSR class
    from backend.main import SubtitleRemover, is_image_file, is_video_or_image
    # Attempt to import config to potentially read defaults if needed, though
    # we rely on SubtitleRemover's internal use of it primarily.
    import backend.config as vsr_config
except ImportError as e:
    print(f"ERROR: Failed to import VSR backend components: {e}", file=sys.stderr)
    print("ERROR: Make sure this script is placed in the root directory of the "
          "'video-subtitle-remover' project and run with its specific Python environment.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during VSR import: {e}", file=sys.stderr)
    sys.exit(1)

def parse_area(area_str: str) -> tuple[int, int, int, int] | None:
    """Parses the area string "ymin,ymax,xmin,xmax" into a tuple."""
    try:
        parts = [int(p.strip()) for p in area_str.split(',')]
        if len(parts) == 4:
            ymin, ymax, xmin, xmax = parts
            # Basic validation (more could be added if needed)
            if ymin < 0 or ymax <= ymin or xmin < 0 or xmax <= xmin:
                raise ValueError("Invalid area coordinates (must be positive and ymax>ymin, xmax>xmin).")
            return ymin, ymax, xmin, xmax
        else:
            raise ValueError("Area string must contain exactly 4 comma-separated integers.")
    except (ValueError, TypeError) as e:
        print(f"ERROR: Invalid format for --area argument '{area_str}'. Expected 'ymin,ymax,xmin,xmax'. Error: {e}", file=sys.stderr)
        return None

def predict_output_path(input_path: str) -> str:
    """Predicts the output path based on VSR's default naming convention."""
    p = Path(input_path)
    input_dir = p.parent
    stem = p.stem
    ext = p.suffix

    if is_image_file(input_path):
        # VSR places processed images in a 'no_sub' subdirectory
        output_dir = input_dir / 'no_sub'
        # Ensure the directory exists (VSR main.py might do this, but good practice here too)
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{stem}{ext}") # Keep original extension for images
    else:
        # VSR places processed videos next to the input with '_no_sub.mp4'
        # NOTE: VSR main.py seems to enforce .mp4 output via VideoWriter.
        return str(input_dir / f"{stem}_no_sub.mp4")


def main():
    parser = argparse.ArgumentParser(description="CLI wrapper for Video Subtitle Remover.")
    parser.add_argument(
        '--input_videos',
        nargs='+',  # Accepts one or more arguments
        required=True,
        help="Path(s) to the input video or image file(s)."
    )
    parser.add_argument(
        '--area',
        type=str,
        required=False,
        default=None,
        help='Optional fixed subtitle area as "ymin,ymax,xmin,xmax". '
             'If not provided, VSR will attempt internal detection (if not skipped in config).'
    )
    # Add other potential arguments here if needed (e.g., overriding config options)

    args = parser.parse_args()

    area_tuple = None
    if args.area:
        area_tuple = parse_area(args.area)
        if area_tuple is None:
            sys.exit(1) # Exit if area format is invalid

    output_paths = []
    print(f"--- Starting VSR CLI Wrapper ---", flush=True)
    print(f"Processing {len(args.input_videos)} file(s)...", flush=True)
    if area_tuple:
        print(f"Using fixed subtitle area: y=[{area_tuple[0]}:{area_tuple[1]}], x=[{area_tuple[2]}:{area_tuple[3]}]", flush=True)
    else:
        # Check VSR config if internal detection is skipped
        detection_skipped = getattr(vsr_config, 'STTN_SKIP_DETECTION', False) and vsr_config.MODE == vsr_config.InpaintMode.STTN
        if detection_skipped:
            print("WARNING: No --area provided, AND internal detection is skipped in VSR config. Inpainting might not occur.", file=sys.stderr, flush=True)
        else:
            print("No fixed subtitle area provided. VSR will use internal detection (if applicable for the chosen mode).", flush=True)


    total_start_time = time.time()

    for i, input_path in enumerate(args.input_videos):
        print(f"\n[{i+1}/{len(args.input_videos)}] Processing: {input_path}", flush=True)
        if not os.path.exists(input_path) or not os.path.isfile(input_path):
            print(f"  ERROR: Input file not found or is not a file. Skipping.", file=sys.stderr, flush=True)
            continue
        if not is_video_or_image(input_path):
             print(f"  ERROR: Input file does not appear to be a supported video or image. Skipping.", file=sys.stderr, flush=True)
             continue

        start_time_single = time.time()
        try:
            # Instantiate SubtitleRemover for the current video
            # gui_mode=False ensures it doesn't expect a GUI environment
            remover = SubtitleRemover(input_path, sub_area=area_tuple, gui_mode=False)

            # Run the VSR processing
            remover.run() # This is a blocking call

            # Predict the output path based on VSR's convention
            predicted_output = predict_output_path(input_path)
            if os.path.exists(predicted_output):
                 output_paths.append(predicted_output)
                 print(f"  SUCCESS: Processing finished. Output: {predicted_output}", flush=True)
            else:
                 # This might happen if VSR failed internally but didn't raise an exception
                 print(f"  ERROR: VSR processing finished but expected output file not found: {predicted_output}", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"  ERROR: VSR processing failed for {input_path}: {e}", file=sys.stderr, flush=True)
            # Optionally print full traceback for debugging
            # import traceback
            # traceback.print_exc()
        finally:
             end_time_single = time.time()
             print(f"  Time taken: {end_time_single - start_time_single:.2f} seconds", flush=True)


    total_end_time = time.time()
    print(f"\n--- VSR CLI Wrapper Finished ---", flush=True)
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds", flush=True)

    # --- Output the list of generated files for the calling process ---
    # Print each successfully generated path, prefixed for easy parsing
    print("\nGenerated Output Files:")
    if output_paths:
        for path in output_paths:
            print(f"OUTPUT_PATH:{path}") # Use a clear prefix
    else:
        print("No output files were successfully generated.")

    # Exit with 0 if at least one file was processed successfully, otherwise 1
    sys.exit(0 if output_paths else 1)


if __name__ == '__main__':
    # It's recommended to use multiprocessing start method 'spawn'
    # This should ideally be set by the environment/caller, but can be forced here
    # Note: VSR's main.py and gui.py also set this.
    import multiprocessing
    try:
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method("spawn", force=True)
            print("Set multiprocessing start method to 'spawn'.")
    except Exception as e:
        print(f"Warning: Could not set start method to 'spawn': {e}")
    main()