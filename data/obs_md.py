import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import statistics


class SharedMetadataManager:
    """Metadata manager using multiprocessing.Manager for shared state"""
    def __init__(self, manager, sampling_rate, integration_time, frequency,
                 fft_size, observation_duration):
        self.shared_dict = manager.dict()
        self.lock = manager.Lock()
        self.metadata_file = None

        # Store configuration
        self.sampling_rate = sampling_rate
        self.integration_time = integration_time
        self.frequency = frequency
        self.fft_size = fft_size
        self.observation_duration = observation_duration

    def initialize_session(self, session_folder, devices):
        """Initialize session metadata"""
        self.metadata_file = os.path.join(session_folder, "observation_metadata.json")
        
        session_start_time = datetime.now(timezone.utc)
        with self.lock:
            self.shared_dict.update({
                "metadata_version": "1.0",
                "created_at": session_start_time.isoformat(),
                "type": "session",
                "configuration": {
                    "sampling_rate": self.sampling_rate,
                    "integration_time": self.integration_time,
                    "center_frequency": self.frequency,
                    "fft_size": self.fft_size,
                    "observation_duration": self.observation_duration
                },
                "session_folder": session_folder,
                "devices": devices,
                "number_of_processes": len(devices),
                "start_time_utc": session_start_time.isoformat(),
                "start_timestamp": session_start_time.timestamp(),
                "end_time_utc": None,
                "end_timestamp": None,
                "duration_seconds": None,
                "status": "started",
                "processes": {}
            })
            
            # Initialize process entries
            processes = {}
            for device in devices:
                processes[device] = {
                    "device": device,
                    "process_id": None,
                    "start_time_utc": None,
                    "end_time_utc": None,
                    "start_timestamp": None,
                    "end_timestamp": None,
                    "duration_seconds": None,
                    "file_size_bytes": None,
                    "total_samples": None,
                    "status": "initialized",
                    "filename": ""
                }
            self.shared_dict["processes"] = processes
        
        self.save_metadata()
    
    def update_process(self, device, **kwargs):
        """Update process-specific metadata"""
        with self.lock:
            processes = dict(self.shared_dict["processes"])
            if device in processes:
                processes[device].update(kwargs)
                if "process_id" not in kwargs:
                    processes[device]["process_id"] = os.getpid()
                self.shared_dict["processes"] = processes
            self.save_metadata()
    
    def finalize_session(self):
        """Finalize session metadata"""
        with self.lock:
            session_end_time = datetime.now(timezone.utc)
            session_start_timestamp = self.shared_dict["start_timestamp"]
            
            # Update session info
            session_data = dict(self.shared_dict)
            session_data.update({
                "end_time_utc": session_end_time.isoformat(),
                "end_timestamp": session_end_time.timestamp(),
                "duration_seconds": session_end_time.timestamp() - session_start_timestamp,
                "status": "completed"
            })
            self.shared_dict.update(session_data)
            self.save_metadata()
    
    def save_metadata(self):
        """Save metadata to file"""
        if self.metadata_file:
            try:
                # Convert shared dict to regular dict for JSON serialization
                metadata_copy = dict(self.shared_dict)
                metadata_copy["processes"] = dict(metadata_copy["processes"])
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata_copy, f, indent=2)
            except Exception as e:
                print(f"Error saving metadata: {e}")
    
    def get_completed_count(self):
        """Get count of completed processes"""
        with self.lock:
            processes = dict(self.shared_dict["processes"])
            completed = sum(1 for p in processes.values() 
                          if p['status'] in ['completed', 'observation_complete'])
            return completed, len(processes)


class ObservationMetadata:
    """
    A comprehensive class for handling paint can interferometer observation metadata.
    Can be created from JSON files and provides useful methods for data analysis.
    """
    
    def __init__(self, metadata_file: Union[str, Path, dict] = None):
        """
        Initialize from a JSON file path or dictionary.
        
        Args:
            metadata_file: Path to JSON file, or dictionary with metadata
        """
        if isinstance(metadata_file, (str, Path)):
            with open(metadata_file, 'r') as f:
                self.data = json.load(f)
        elif isinstance(metadata_file, dict):
            self.data = metadata_file
        else:
            raise ValueError("metadata_file must be a file path or dictionary")
        
        self._validate_metadata()

    def __str__(self) -> str:
        return self.summary()
    
    def _validate_metadata(self):
        """Validate that the metadata has the expected structure"""
        required_fields = ["type", "configuration", "processes"]
        for field in required_fields:
            if field not in self.data:
                raise ValueError(f"Missing required field: {field}")
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]):
        """Create instance from JSON file"""
        return cls(filepath)
    
    @classmethod
    def from_dict(cls, metadata_dict: dict):
        """Create instance from dictionary"""
        return cls(metadata_dict)
    
    # ==================== BASIC PROPERTIES ====================
    
    @property
    def metadata_version(self) -> str:
        """Get metadata version"""
        return self.data.get("metadata_version", "unknown")
    
    @property
    def session_folder(self) -> str:
        """Get session folder name"""
        return self.data.get("session_folder", "")
    
    @property
    def observation_type(self) -> str:
        """Get observation type"""
        return self.data.get("type", "unknown")
    
    @property
    def status(self) -> str:
        """Get overall session status"""
        return self.data.get("status", "unknown")
    
    # ==================== CONFIGURATION ====================
    
    @property
    def configuration(self) -> dict:
        """Get full configuration"""
        return self.data.get("configuration", {})
    
    @property
    def sampling_rate(self) -> float:
        """Get sampling rate in Hz"""
        sr = self.configuration.get("sampling_rate", 0)
        return float(sr) if isinstance(sr, str) else sr
    
    @property
    def center_frequency(self) -> float:
        """Get center frequency in Hz"""
        cf = self.configuration.get("center_frequency", 0)
        return float(cf) if isinstance(cf, str) else cf
    
    @property
    def integration_time(self) -> int:
        """Get integration time in seconds"""
        return self.configuration.get("integration_time", 0)
    
    @property
    def fft_size(self) -> int:
        """Get FFT size"""
        return self.configuration.get("fft_size", 0)
    
    @property
    def observation_duration(self) -> int:
        """Get planned observation duration in seconds"""
        return self.configuration.get("observation_duration", 0)
    
    # ==================== TIMING ====================
    
    @property
    def session_start_time(self) -> Optional[datetime]:
        """Get session start time as datetime object"""
        start_str = self.data.get("start_time_utc")
        return datetime.fromisoformat(start_str) if start_str else None
    
    @property
    def session_end_time(self) -> Optional[datetime]:
        """Get session end time as datetime object"""
        end_str = self.data.get("end_time_utc")
        return datetime.fromisoformat(end_str) if end_str else None
    
    @property
    def session_duration(self) -> float:
        """Get actual session duration in seconds"""
        return self.data.get("duration_seconds", 0.0)
    
    @property
    def session_start_timestamp(self) -> float:
        """Get session start timestamp"""
        return self.data.get("start_timestamp", 0.0)
    
    @property
    def session_end_timestamp(self) -> float:
        """Get session end timestamp"""
        return self.data.get("end_timestamp", 0.0)
    
    # ==================== DEVICE/PROCESS MANAGEMENT ====================
    
    def list_devices(self) -> List[str]:
        """Get list of all devices in the observation"""
        return list(self.data.get("processes", {}).keys())
    
    @property
    def device_count(self) -> int:
        """Get number of devices"""
        return len(self.list_devices())
    
    def get_process(self, device: str) -> Optional[dict]:
        """Get process information for a specific device"""
        return self.data.get("processes", {}).get(device)
    
    def get_processes(self, devices: List[str] = None) -> Dict[str, dict]:
        """
        Get process information for specified devices or all devices
        
        Args:
            devices: List of device names, or None for all devices
        """
        all_processes = self.data.get("processes", {})
        if devices is None:
            return all_processes
        return {dev: all_processes[dev] for dev in devices if dev in all_processes}
    
    def get_processes_by_status(self, status: str) -> Dict[str, dict]:
        """Get all processes with a specific status"""
        return {
            device: process for device, process in self.data.get("processes", {}).items()
            if process.get("status") == status
        }
    
    # ==================== PROCESS TIMING ====================
    
    def get_process_start_time(self, device: str) -> Optional[datetime]:
        """Get process start time for a device"""
        process = self.get_process(device)
        if process and process.get("start_time_utc"):
            return datetime.fromisoformat(process["start_time_utc"])
        return None
    
    def get_process_end_time(self, device: str) -> Optional[datetime]:
        """Get process end time for a device"""
        process = self.get_process(device)
        if process and process.get("end_time_utc"):
            return datetime.fromisoformat(process["end_time_utc"])
        return None
    
    def get_process_duration(self, device: str) -> float:
        """Get process duration for a device in seconds"""
        process = self.get_process(device)
        return process.get("duration_seconds", 0.0) if process else 0.0
    
    def get_all_process_durations(self) -> Dict[str, float]:
        """Get durations for all processes"""
        return {
            device: process.get("duration_seconds", 0.0)
            for device, process in self.data.get("processes", {}).items()
        }
    
    # ==================== DATA ANALYSIS ====================
    
    def get_file_info(self, device: str) -> Tuple[Optional[str], int, int]:
        """
        Get file information for a device
        
        Returns:
            Tuple of (filename, file_size_bytes, total_samples)
        """
        process = self.get_process(device)
        if process:
            return (
                process.get("filename"),
                process.get("file_size_bytes", 0),
                process.get("total_samples", 0)
            )
        return None, 0, 0
    
    def get_all_file_sizes(self) -> Dict[str, int]:
        """Get file sizes for all devices"""
        return {
            device: process.get("file_size_bytes", 0)
            for device, process in self.data.get("processes", {}).items()
        }
    
    def get_all_sample_counts(self) -> Dict[str, int]:
        """Get sample counts for all devices"""
        return {
            device: process.get("total_samples", 0)
            for device, process in self.data.get("processes", {}).items()
        }
    
    def are_files_synchronized(self) -> bool:
        """Check if all files have the same size (synchronized)"""
        sizes = list(self.get_all_file_sizes().values())
        return len(set(sizes)) <= 1 if sizes else True
    
    # ==================== STATISTICS ====================
    
    def get_timing_statistics(self) -> dict:
        """Get timing statistics for all processes"""
        durations = list(self.get_all_process_durations().values())
        if not durations:
            return {}
        
        return {
            "mean_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "duration_range": max(durations) - min(durations)
        }
    
    def get_completion_summary(self) -> dict:
        """Get summary of process completion status"""
        processes = self.data.get("processes", {})
        status_counts = {}
        
        for process in processes.values():
            status = process.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total = len(processes)
        completed = status_counts.get("observation_complete", 0) + status_counts.get("completed", 0)
        
        return {
            "total_processes": total,
            "completed_processes": completed,
            "completion_rate": completed / total if total > 0 else 0.0,
            "status_breakdown": status_counts
        }
    
    # ==================== UTILITY METHODS ====================
    
    def get_earliest_start_time(self) -> Optional[datetime]:
        """Get the earliest process start time"""
        start_times = []
        for process in self.data.get("processes", {}).values():
            if process.get("start_time_utc"):
                start_times.append(datetime.fromisoformat(process["start_time_utc"]))
        return min(start_times) if start_times else None
    
    def get_latest_end_time(self) -> Optional[datetime]:
        """Get the latest process end time"""
        end_times = []
        for process in self.data.get("processes", {}).values():
            if process.get("end_time_utc"):
                end_times.append(datetime.fromisoformat(process["end_time_utc"]))
        return max(end_times) if end_times else None
    
    def get_process_overlap_duration(self) -> float:
        """Get the duration when all processes were running simultaneously"""
        latest_start = self.get_earliest_start_time()
        earliest_end = self.get_latest_end_time()
        
        if latest_start and earliest_end:
            # Find latest start among all processes
            for process in self.data.get("processes", {}).values():
                if process.get("start_time_utc"):
                    start_time = datetime.fromisoformat(process["start_time_utc"])
                    if start_time > latest_start:
                        latest_start = start_time
            
            # Find earliest end among all processes  
            for process in self.data.get("processes", {}).values():
                if process.get("end_time_utc"):
                    end_time = datetime.fromisoformat(process["end_time_utc"])
                    if end_time < earliest_end:
                        earliest_end = end_time
            
            overlap = (earliest_end - latest_start).total_seconds()
            return max(0.0, overlap)
        
        return 0.0
    
    def to_dict(self) -> dict:
        """Get the raw metadata dictionary"""
        return self.data.copy()
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save metadata to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def summary(self) -> str:
        """Get a human-readable summary of the observation"""
        completion = self.get_completion_summary()
        timing_stats = self.get_timing_statistics()
        
        summary_lines = [
            f"=== Observation Summary ===",
            f"Session: {self.session_folder}",
            f"Status: {self.status}",
            f"Devices: {self.device_count}",
            f"Session Duration: {self.session_duration:.2f}s",
            f"Completion Rate: {completion['completion_rate']*100:.1f}%",
            f"Synchronized Files: {'Yes' if self.are_files_synchronized() else 'No'}",
            f"",
            f"=== Configuration ===",
            f"Sampling Rate: {self.sampling_rate:,.0f} Hz",
            f"Center Frequency: {self.center_frequency:,.0f} Hz", 
            f"FFT Size: {self.fft_size}",
            f"Integration Time: {self.integration_time}s",
            f"Planned Duration: {self.observation_duration}s",
        ]
        
        if timing_stats:
            summary_lines.extend([
                f"",
                f"=== Process Timing ===",
                f"Mean Duration: {timing_stats['mean_duration']:.2f}s",
                f"Duration Range: {timing_stats['duration_range']:.2f}s",
                f"Process Overlap: {self.get_process_overlap_duration():.2f}s"
            ])
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return f"ObservationMetadata(session='{self.session_folder}', devices={self.device_count}, status='{self.status}')"
    