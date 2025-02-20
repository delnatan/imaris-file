from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imaris_reader import ImarisReader


def basic_metadata_inspection(file_path: str) -> None:
    """Example of inspecting metadata from an Imaris file."""
    with ImarisReader(file_path) as reader:
        # Print basic image information
        img_meta = reader.image_metadata
        print(f"Image size (X,Y,Z): {img_meta.size}")
        print(f"Pixel size: {img_meta.pixel_sizes} {img_meta.unit}")

        # Print channel information
        for i, channel in enumerate(reader.channel_metadata):
            print(f"\nChannel {i}: {channel.name}")
            if channel.emission_wavelength:
                print(f"  Emission λ: {channel.emission_wavelength}nm")
            if channel.excitation_wavelength:
                print(f"  Excitation λ: {channel.excitation_wavelength}nm")

        # Print time information
        time_meta = reader.time_metadata
        print(f"\nNumber of timepoints: {time_meta.num_timepoints}")
        if time_meta.interval:
            print(f"Time interval: {time_meta.interval}s")


def maximum_intensity_projection(file_path: str, output_dir: Path) -> None:
    """Create maximum intensity projections for each channel."""
    with ImarisReader(file_path) as reader:
        output_dir.mkdir(exist_ok=True)

        for ch_idx, channel in enumerate(reader.channel_metadata):
            # Read the full z-stack for this channel
            data = reader.get_data(channel=ch_idx)

            # Create maximum intensity projection
            mip = np.max(data, axis=0)

            # Simple auto-scaling for display
            p2, p98 = np.percentile(mip, (2, 98))
            scaled = np.clip(mip, p2, p98)
            scaled = (scaled - p2) / (p98 - p2)

            # Save the projection
            plt.figure(figsize=(10, 10))
            plt.imshow(scaled, cmap="gray")
            plt.title(f"MIP - {channel.name}")
            plt.colorbar()
            plt.axis("off")
            plt.savefig(
                output_dir / f"mip_channel_{ch_idx}_{channel.name}.png"
            )
            plt.close()


def process_large_image_in_chunks(file_path: str) -> np.ndarray:
    """Example of processing a large image in chunks."""
    with ImarisReader(file_path) as reader:
        # Get image dimensions
        img_meta = reader.image_metadata

        # Estimate memory usage for full volume
        mem_usage = reader.estimate_memory_usage(
            time_points=0, channels=0, resolution_level=0
        )
        print(f"Full volume would use {mem_usage / 1e9:.2f} GB of memory")

        # Create output array for processed data
        output = np.zeros(img_meta.size, dtype=np.float32)

        # Process chunks with overlap to avoid edge effects
        chunk_size = (32, 256, 256)
        overlap = (4, 16, 16)

        for (z_slice, y_slice, x_slice), chunk_data in reader.iterate_chunks(
            chunk_size=chunk_size,
            overlap=overlap,
            channels=0,  # Just process first channel
        ):
            # Example processing: Gaussian filter
            from scipy.ndimage import gaussian_filter

            processed_chunk = gaussian_filter(chunk_data, sigma=2)

            # Remove overlap regions for saving
            z_start = (
                z_slice.start + overlap[0]
                if z_slice.start > 0
                else z_slice.start
            )
            z_end = (
                z_slice.stop - overlap[0]
                if z_slice.stop < img_meta.size[2]
                else z_slice.stop
            )
            y_start = (
                y_slice.start + overlap[1]
                if y_slice.start > 0
                else y_slice.start
            )
            y_end = (
                y_slice.stop - overlap[1]
                if y_slice.stop < img_meta.size[1]
                else y_slice.stop
            )
            x_start = (
                x_slice.start + overlap[2]
                if x_slice.start > 0
                else x_slice.start
            )
            x_end = (
                x_slice.stop - overlap[2]
                if x_slice.stop < img_meta.size[0]
                else x_slice.stop
            )

            # Save processed data
            output[z_start:z_end, y_start:y_end, x_start:x_end] = (
                processed_chunk[
                    overlap[0] : -overlap[0] if overlap[0] > 0 else None,
                    overlap[1] : -overlap[1] if overlap[1] > 0 else None,
                    overlap[2] : -overlap[2] if overlap[2] > 0 else None,
                ]
            )

        return output


def analyze_time_series(file_path: str) -> dict:
    """Analyze intensity changes over time."""
    with ImarisReader(file_path) as reader:
        time_meta = reader.time_metadata
        results = {}

        for ch_idx, channel in enumerate(reader.channel_metadata):
            mean_intensities = []
            max_intensities = []

            for t in range(time_meta.num_timepoints):
                data = reader.get_data(time_point=t, channel=ch_idx)
                mean_intensities.append(np.mean(data))
                max_intensities.append(np.max(data))

            results[channel.name] = {
                "mean_intensity": mean_intensities,
                "max_intensity": max_intensities,
                "timepoints": time_meta.timepoints,
            }

        return results


if __name__ == "__main__":
    # Example usage
    file_path = "path/to/your/file.ims"
    output_dir = Path("output")

    print("Inspecting metadata...")
    basic_metadata_inspection(file_path)

    print("\nCreating maximum intensity projections...")
    maximum_intensity_projection(file_path, output_dir)

    print("\nProcessing large image in chunks...")
    processed_data = process_large_image_in_chunks(file_path)

    print("\nAnalyzing time series...")
    time_analysis = analyze_time_series(file_path)

    # Plot time series results
    for channel, data in time_analysis.items():
        plt.figure(figsize=(10, 5))
        plt.plot(data["timepoints"], data["mean_intensity"], label="Mean")
        plt.plot(data["timepoints"], data["max_intensity"], label="Max")
        plt.title(f"Intensity over time - {channel}")
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.legend()
        plt.savefig(output_dir / f"time_series_{channel}.png")
        plt.close()
