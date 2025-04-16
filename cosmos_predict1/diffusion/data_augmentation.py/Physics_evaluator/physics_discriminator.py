# Import Neccessary Libraries here Complete this
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import cv2
from tqdm import tqdm

from YOLO_extractor import YOLO_ADAE
import argparse
from physics_utils import load_from_fileobj


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the physics discriminator.
    
    This function configures and processes command line arguments for the YOLO-based 
    physics discriminator. It handles model selection, video paths, batch processing 
    options, and result saving preferences.
    
    Returns:
        argparse.Namespace: A namespace object containing all parsed arguments with the following attributes:
            - model (str): YOLO model version to use (default: 'yolo12x-cls.pt')
            - input (str): Path to the input reference video
            - output (str): Path to the output video to be analyzed
            - batch (str, optional): Path to YAML file containing multiple output videos for batch processing
            - save_csv (bool): Flag to enable saving results to CSV (default: False)
            - results_folder (str): Directory to save analysis results (default: 'results')
    
    Example:
        To analyze a single video:
        ```
        python physics_discriminator.py --model yolov8m.pt --input reference.mp4 --output test.mp4 --save_csv
        ```
        
        For batch processing:
        ```
        python physics_discriminator.py --input reference.mp4 --batch videos.yaml --results_folder results_batch
        ```
    """
    parser = argparse.ArgumentParser(description='Physics Discriminator using YOLO feature extraction')
    
    # Model name
    parser.add_argument('--model', type=str, default='yolo12x-cls.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                                'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
                                'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt',
                                'yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt', 'yolov8l-cls.pt', 'yolov8x-cls.pt',
                                'yolo11n-cls.pt', 'yolo11s-cls.pt', 'yolo11m-cls.pt', 'yolo11l-cls.pt', 'yolo11x-cls.pt',
                                'yolo12n-cls.pt', 'yolo12s-cls.pt', 'yolo12m-cls.pt', 'yolo12l-cls.pt', 'yolo12x-cls.pt'],
                        help='YOLO model name (default: yolo12x-cls.pt)')
    
    # Input and output video paths
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input video')
    parser.add_argument('--output', type=str, required=True,
                        help='Path for output video')
    
    # Batch processing
    parser.add_argument('--batch', type=str, default=None,
                        help='YAML file path for list of batch output videos')
    
    # Results saving
    parser.add_argument('--save_csv', action='store_true',
                        help='Save results to CSV file')
    parser.add_argument('--results_folder', type=str, default='results',
                        help='Output folder to save results (default: results)')
    
    return parser.parse_args()


# write the function to calculate rate of change of cosine similarity
def calculate_rate_of_change(similarity_values, time_frames=None):
    """
    Calculate the rate of change of cosine similarity over time/frames
    
    Args:
        similarity_values: List or array of cosine similarity values
        time_frames: List or array of time/frame indices (default: None, uses frame indices)
        
    Returns:
        rate_of_change: Array of rate of change values
    """
    similarity_values = np.array(similarity_values)
    
    # If time_frames not provided, use frame indices
    if time_frames is None:
        time_frames = np.arange(len(similarity_values))
    else:
        time_frames = np.array(time_frames)
    
    # Calculate difference between consecutive similarity values
    similarity_diff = np.diff(similarity_values)
    
    # Calculate difference between consecutive time frames
    time_diff = np.diff(time_frames)
    
    # Calculate rate of change (derivative)
    rate_of_change = similarity_diff / time_diff
    
    # Add a zero at the beginning to maintain the same array length (optional)
    # This makes plotting easier by aligning with original frames
    rate_of_change = np.insert(rate_of_change, 0, 0)
    
    return rate_of_change


def main(args):
    """Execute the physics discriminator pipeline to analyze physical consistency in videos.
    
    This function implements a complete analysis pipeline that:
    1. Processes reference and test videos using the YOLO feature extractor
    2. Extracts embeddings from video frames and computes similarity metrics
    3. Calculates the rate of change of similarity over time 
    4. Visualizes results and saves data for further analysis
    
    The pipeline can process either a single video or multiple videos in batch mode.
    It uses the last frame of the input video as a reference and compares all frames
    of the output video(s) against this reference.
    
    Args:
        args (argparse.Namespace): Command line arguments containing:
            - model: YOLO model specification
            - input: Path to reference video
            - output: Path to test video 
            - batch: Optional path to YAML file with multiple test videos
            - save_csv: Flag to save numerical results
            - results_folder: Directory for saving outputs
    
    Returns:
        None: Results are saved to disk as plots and optional CSV files
        
    Notes:
        - The cosine similarity measures feature-space similarity between frames
        - The rate of change of similarity can identify physically inconsistent transitions
        - For each output video, both metrics are plotted against time
        - Detection results from the YOLO model are also recorded
    """
    # Create results directory if it doesn't exist
    os.makedirs(args.results_folder, exist_ok=True)
    
    # Initialize YOLO model
    print(f"Loading YOLO model: {args.model}")
    yolo_model = YOLO_ADAE(model_version=args.model)
    
    # Load input video
    print(f"Loading input video: {args.input}")
    input_frames, input_metadata = load_from_fileobj(args.input)
    
    # Get the last frame of the input video for reference
    reference_frame = input_frames[-1]
    
    # Run YOLO inference on the reference frame
    reference_results = yolo_model.run_inference(reference_frame)
    reference_prediction = reference_results[0].boxes.data.cpu().numpy() if reference_results else None
    
    # Extract features from the reference frame
    reference_embedding = yolo_model.extract_features(reference_frame)
    
    # Initialize variables for batch processing
    batch_videos = []
    
    # Check if batch processing is required
    if args.batch and os.path.exists(args.batch):
        print(f"Loading batch videos from: {args.batch}")
        with open(args.batch, 'r') as file:
            batch_config = yaml.safe_load(file)
            batch_videos = batch_config.get('videos', [])
    else:
        # Single video processing
        batch_videos = [args.output]
        
    # Create figure for plotting
    plt.figure(figsize=(12, 8))
    
    # Initialize dictionary to store results for CSV
    results_data = {
        'video_name': [],
        'frame_number': [],
        'cosine_similarity': [],
        'rate_of_change': [],
        'has_detection': []
    }
    
    # Process each video
    for video_idx, video_path in enumerate(batch_videos):
        print(f"Processing video {video_idx + 1}/{len(batch_videos)}: {video_path}")
        
        # Load output video
        output_frames, output_metadata = load_from_fileobj(video_path)
        
        # Initialize arrays to store results
        cosine_similarities = []
        predictions = []
        
        # Process each frame of the output video
        for frame_idx, frame in enumerate(tqdm(output_frames, desc="Processing frames")):
            # Extract features from the current frame
            current_embedding = yolo_model.extract_features(frame)
            
            # Calculate cosine similarity with the reference embedding
            similarity = yolo_model.compute_similarity(reference_embedding, current_embedding)
            cosine_similarities.append(similarity)
            
            # Run YOLO inference to get predictions
            results = yolo_model.run_inference(frame)
            predictions.append(results[0].boxes.data.cpu().numpy() if results else None)
            
            # Store data for CSV
            video_name = os.path.basename(video_path)
            results_data['video_name'].append(video_name)
            results_data['frame_number'].append(frame_idx)
            results_data['cosine_similarity'].append(similarity)
            results_data['has_detection'].append(bool(results and len(results[0].boxes.data) > 0))
        
        # Calculate rate of change of cosine similarity
        fps = output_metadata.get('fps', 30)
        time_frames = np.arange(len(cosine_similarities)) / fps  # Convert to seconds
        rate_of_change = calculate_rate_of_change(cosine_similarities, time_frames)
        
        # Add rate of change to results data
        for i, roc in enumerate(rate_of_change):
            if i < len(results_data['rate_of_change']):
                results_data['rate_of_change'].append(roc)
        
        # Plot cosine similarity and rate of change
        plt.subplot(2, 1, 1)
        plt.plot(time_frames, cosine_similarities, label=f'Video {video_idx + 1}: {os.path.basename(video_path)}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cosine Similarity')
        plt.title('Cosine Similarity over Time')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(time_frames, rate_of_change, label=f'Video {video_idx + 1}: {os.path.basename(video_path)}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Rate of Change')
        plt.title('Rate of Change of Cosine Similarity over Time')
        plt.grid(True)
        plt.legend()
    
    # Save the plot
    plot_path = os.path.join(args.results_folder, 'similarity_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Save results to CSV if required
    if args.save_csv:
        csv_path = os.path.join(args.results_folder, 'similarity_results.csv')
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # Show plot (optional)
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)