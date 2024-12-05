from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np


@dataclass
class ImageMetadata:
    size: Tuple[int, int, int]  # x, y, z
    unit: str
    description: str
    recording_date: str
    pixel_sizes: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    extent: Tuple[float, float, float]
    numerical_aperture: float
    microscope_mode: str
    lens_power: int


@dataclass
class ChannelMetadata:
    name: str
    description: str
    emission_wavelength: Optional[float]
    excitation_wavelength: Optional[float]


@dataclass
class TimeMetadata:
    num_timepoints: int
    timepoints: List[datetime]
    interval: Optional[float]


def _pprint_meta(meta: dataclass):
    for field in fields(meta):
        _val = getattr(meta, field.name)
        print(f"\t{field.name} : {_val}")


class ImarisReader:
    """Reader for Imaris (HDF5) microscopy files with efficient data access
    patterns.

    """

    def __init__(self, file_path: str, validate: bool = True):
        """
        Initialize the Imaris reader.

        Args:
            file_path: Path to the Imaris file
            validate: Whether to validate the file structure

        Raises:
            ValueError: If the file is not a valid Imaris file
            IOError: If the file cannot be opened
        """
        self.file_path = file_path
        self.file: Optional[h5py.File] = None
        self._image_meta: Optional[ImageMetadata] = None
        self._channel_meta: Optional[List[ChannelMetadata]] = None
        self._time_meta: Optional[TimeMetadata] = None

        try:
            self._open_file()
            if validate:
                self._validate_file_structure()
            self._read_metadata()
        except Exception as e:
            self.close()
            raise IOError(
                f"Failed to initialize Imaris reader: {str(e)}"
            ) from e

    def _validate_file_structure(self) -> None:
        """Validate the basic structure of the Imaris file."""
        required_groups = ["DataSet", "DataSetInfo"]
        for group in required_groups:
            if group not in self.file:
                raise ValueError(f"Invalid Imaris file: missing {group} group")

    def _open_file(self) -> None:
        """Open the HDF5 file for reading."""
        try:
            self.file = h5py.File(self.file_path, "r")
        except Exception as e:
            raise IOError(
                f"Failed to open file {self.file_path}: {str(e)}"
            ) from e

    def _decode_attr(self, attr: Union[bytes, np.ndarray, Any]) -> Any:
        """Decode HDF5 attributes to Python types."""
        if isinstance(attr, bytes):
            return attr.decode("utf-8")
        elif isinstance(attr, np.ndarray) and attr.dtype.kind == "S":
            return attr.astype(str)
        return attr

    def _read_attrs(self, group: h5py.Group) -> dict:
        """Read and decode all attributes from an HDF5 group."""
        attrs = {}

        for key, value in group.attrs.items():
            decoded_value = self._decode_attr(value)

            if isinstance(decoded_value, np.ndarray):
                if decoded_value.dtype.kind in ["U", "S"]:
                    attrs[key] = "".join(decoded_value)
                elif decoded_value.size == 1:
                    attrs[key] = decoded_value.item()
                else:
                    attrs[key] = decoded_value.tolist()
            else:
                attrs[key] = decoded_value

            # Convert to numeric type if possible
            if isinstance(attrs[key], str):
                try:
                    if "." in attrs[key]:
                        attrs[key] = float(attrs[key])
                    else:
                        attrs[key] = int(attrs[key])
                except ValueError:
                    pass  # Keep as string if conversion fails

        return attrs

    def _count_channels(self) -> int:
        """Count the number of channels in the dataset."""
        dataset_info = self.file["DataSetInfo"]
        return sum(
            1 for key in dataset_info.keys() if key.startswith("Channel ")
        )

    def _parse_timepoints(self, time_info_attrs: dict) -> List[datetime]:
        """Parse timepoint strings into datetime objects."""
        timepoints = []
        i = 1
        while f"TimePoint{i}" in time_info_attrs:
            timestamp_str = time_info_attrs[f"TimePoint{i}"]
            try:
                timestamp = datetime.strptime(
                    timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                )
                timepoints.append(timestamp)
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse timestamp for TimePoint{i}: {timestamp_str}"
                ) from e
            i += 1
        return timepoints

    def _calculate_intervals(
        self, timepoints: List[datetime]
    ) -> Optional[float]:
        """Calculate average time interval between timepoints."""
        if len(timepoints) < 2:
            return None

        intervals = [
            (timepoints[i] - timepoints[i - 1]).total_seconds()
            for i in range(1, len(timepoints))
        ]
        avg_interval = sum(intervals) / len(intervals)
        return round(avg_interval, 3)

    def _read_metadata(self) -> None:
        """Read and parse all metadata from the file."""
        if self.file is None:
            raise RuntimeError("File not opened")

        dataset_info = self.file["DataSetInfo"]

        # Read image metadata
        image = dataset_info["Image"]
        image_attrs = self._read_attrs(image)

        # Calculate pixel sizes
        origin = (
            image_attrs["ExtMin0"],
            image_attrs["ExtMin1"],
            image_attrs["ExtMin2"],
        )
        extent = (
            image_attrs["ExtMax0"],
            image_attrs["ExtMax1"],
            image_attrs["ExtMax2"],
        )
        pixel_sizes = (
            round((extent[0] - origin[0]) / image_attrs["X"], 4),
            round((extent[1] - origin[1]) / image_attrs["Y"], 4),
            round((extent[2] - origin[2]) / image_attrs["Z"], 4),
        )

        # Create ImageMetadata instance
        self._image_meta = ImageMetadata(
            size=(image_attrs["X"], image_attrs["Y"], image_attrs["Z"]),
            unit=image_attrs["Unit"],
            description=image_attrs["Description"],
            recording_date=image_attrs["RecordingDate"],
            pixel_sizes=pixel_sizes,
            origin=origin,
            extent=extent,
            numerical_aperture=image_attrs["NumericalAperture"],
            microscope_mode=image_attrs["MicroscopeMode"],
            lens_power=image_attrs["LensPower"],
        )

        # Read channel metadata
        channels = []
        for i in range(self._count_channels()):
            channel = dataset_info[f"Channel {i}"]
            channel_attrs = self._read_attrs(channel)
            channels.append(
                ChannelMetadata(
                    name=channel_attrs["Name"],
                    description=channel_attrs["Description"],
                    emission_wavelength=channel_attrs.get(
                        "LSMEmissionWavelength"
                    ),
                    excitation_wavelength=channel_attrs.get(
                        "LSMExcitationWavelength"
                    ),
                )
            )
        self._channel_meta = channels

        # Read time metadata
        time_info = dataset_info["TimeInfo"]
        time_info_attrs = self._read_attrs(time_info)
        timepoints = self._parse_timepoints(time_info_attrs)

        self._time_meta = TimeMetadata(
            num_timepoints=time_info_attrs["DatasetTimePoints"],
            timepoints=timepoints,
            interval=self._calculate_intervals(timepoints),
        )

    @property
    def image_metadata(self) -> ImageMetadata:
        """Get image metadata."""
        if self._image_meta is None:
            raise RuntimeError("Metadata not initialized")
        return self._image_meta

    @property
    def channel_metadata(self) -> List[ChannelMetadata]:
        """Get channel metadata."""
        if self._channel_meta is None:
            raise RuntimeError("Metadata not initialized")
        return self._channel_meta

    @property
    def time_metadata(self) -> TimeMetadata:
        """Get time series metadata."""
        if self._time_meta is None:
            raise RuntimeError("Metadata not initialized")
        return self._time_meta

    def get_resolution_levels(self) -> List[int]:
        """Get available resolution levels."""
        return sorted(
            int(k.split()[-1])
            for k in self.file["DataSet"].keys()
            if k.startswith("ResolutionLevel")
        )

    def get_data(
        self,
        *,  # Force keyword arguments
        time_point: int = 0,
        channel: Optional[Union[int, List[int]]] = None,
        resolution_level: int = 0,
        roi: Optional[Tuple[slice, slice, slice]] = None,
        out_dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Get image data with flexible access patterns.

        Args:
            time_point: Time point index
            channel: Channel index or list of indices. None means all channels
            resolution_level: Resolution level to read from
            roi: Region of interest as (z_slice, y_slice, x_slice)
            out_dtype: Output data type. None means keep original

        Returns:
            Image data as numpy array with shape (C, Z, Y, X) or (Z, Y, X)
        """
        if channel is None:
            channel = list(range(len(self.channel_metadata)))
        channels = [channel] if isinstance(channel, int) else channel

        # Build ROI if not provided
        if roi is None:
            roi = (
                slice(None, self.image_metadata.size[2]),  # Z
                slice(None, self.image_metadata.size[1]),  # Y
                slice(None, self.image_metadata.size[0]),  # X
            )

        # Read data for each channel
        data_list = []
        for ch in channels:
            dataset_path = (
                f"/DataSet/ResolutionLevel {resolution_level}/"
                f"TimePoint {time_point}/Channel {ch}/Data"
            )
            data = self.file[dataset_path][roi]
            if out_dtype:
                data = data.astype(out_dtype)
            data_list.append(data)

        # Stack if multiple channels
        if len(data_list) > 1:
            return np.stack(data_list)
        return data_list[0]

    def iterate_chunks(
        self,
        chunk_size: Tuple[int, int, int],
        *,  # Force keyword arguments
        time_point: int = 0,
        channels: Optional[List[int]] = None,
        overlap: Tuple[int, int, int] = (0, 0, 0),
        roi: Optional[Tuple[slice, slice, slice]] = None,
    ) -> Iterator[Tuple[Tuple[slice, slice, slice], np.ndarray]]:
        """
        Iterate over chunks of the image with optional overlap.

        Args:
            chunk_size: Size of chunks (Z, Y, X)
            time_point: Time point to read from
            channels: List of channels to read. None means all channels
            overlap: Overlap between chunks (Z, Y, X)
            roi: Region of interest to iterate over

        Yields:
            Tuple of (chunk_slices, chunk_data)
        """
        if channels is None:
            channels = list(range(len(self.channel_metadata)))

        # Calculate iteration bounds
        if roi is None:
            bounds = (
                (0, self.image_metadata.size[2]),  # Z
                (0, self.image_metadata.size[1]),  # Y
                (0, self.image_metadata.size[0]),  # X
            )
        else:
            bounds = (
                (
                    roi[0].start or 0,
                    roi[0].stop or self.image_metadata.size[2],
                ),
                (
                    roi[1].start or 0,
                    roi[1].stop or self.image_metadata.size[1],
                ),
                (
                    roi[2].start or 0,
                    roi[2].stop or self.image_metadata.size[0],
                ),
            )

        # Calculate steps
        steps = tuple(max(1, c - o) for c, o in zip(chunk_size, overlap))

        # Iterate over chunks
        for z in range(bounds[0][0], bounds[0][1], steps[0]):
            for y in range(bounds[1][0], bounds[1][1], steps[1]):
                for x in range(bounds[2][0], bounds[2][1], steps[2]):
                    # Calculate chunk bounds
                    chunk_slices = (
                        slice(z, min(z + chunk_size[0], bounds[0][1])),
                        slice(y, min(y + chunk_size[1], bounds[1][1])),
                        slice(x, min(x + chunk_size[2], bounds[2][1])),
                    )

                    # Get data for this chunk
                    chunk_data = self.get_data(
                        time_point=time_point,
                        channel=channels,
                        roi=chunk_slices,
                    )

                    yield chunk_slices, chunk_data

    def estimate_memory_usage(
        self,
        time_points: Union[int, List[int]],
        channels: Union[int, List[int]],
        resolution_level: int = 0,
    ) -> int:
        """Estimate memory usage for a given data access pattern in bytes."""
        if isinstance(time_points, int):
            time_points = [time_points]
        if isinstance(channels, int):
            channels = [channels]

        sample_data = self.get_data(
            time_point=time_points[0],
            channel=channels[0],
            resolution_level=resolution_level,
        )

        return sample_data.nbytes * len(time_points) * len(channels)

    def info(self):
        """prints out metadata of Imaris file"""
        print("Image info:")
        _pprint_meta(self.image_metadata)
        print("Channel info:")
        for i, ch_meta in enumerate(self.channel_metadata):
            print(f"Channel index = {i}")
            _pprint_meta(ch_meta)
        print("Timelapse info:")
        _pprint_meta(self.time_metadata)

    def close(self) -> None:
        """Close the file handle."""
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self) -> "ImarisReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
