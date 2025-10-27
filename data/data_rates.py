'''
Module for calculating data rates and total data volumes with unit conversions.

Defines functions to compute data rates based on sampling parameters
and to calculate total data volumes over specified observation times.
It includes a DataUnit class for handling unit conversions between
various data rate and volume units.
'''
from typing import Union
import warnings


class DataUnit:
    '''
    Class to represent data rates and volumes with unit conversions.
    Supported units:
    Units for data rates:
        - bytes/s
        - kb/s
        - mb/s
        - gb/s
        - bytes/min
        - kb/min
        - mb/min
        - gb/min
        - bytes/h
        - kb/h
        - mb/h
        - gb/h
    Units for data volumes:
        - bytes
        - kb
        - mb
        - gb
    '''
    BYTES_PER_SECOND = "bytes/s"
    KILOBYTES_PER_SECOND = "kb/s"
    MEGABYTES_PER_SECOND = "mb/s"
    GIGABYTES_PER_SECOND = "gb/s"
    BYTES_PER_HOUR = "bytes/h"
    KILOBYTES_PER_HOUR = "kb/h"
    MEGABYTES_PER_HOUR = "mb/h"
    GIGABYTES_PER_HOUR = "gb/h"
    BYTES_PER_MINUTE = "bytes/min"
    KILOBYTES_PER_MINUTE = "kb/min"
    MEGABYTES_PER_MINUTE = "mb/min"
    GIGABYTES_PER_MINUTE = "gb/min"
    BYTES = "bytes"
    KILOBYTES = "kb"
    MEGABYTES = "mb"
    GIGABYTES = "gb"

    def __init__(self, value: float, unit: str):
        """
        Initialize a DataUnit instance.

        :value: numerical value of the data rate or volume
        :type value: float
        :unit: unit of the data rate or volume
        :type unit: str
        """
        unit = unit.lower()
        if not DataUnit.is_valid_unit(unit):
            raise ValueError(f"Invalid unit '{unit}'. Valid units are: {DataUnit.all_units()}")
        self.value = value
        self.unit = unit

    def __str__(self) -> str:
        """
        String representation of the DataUnit

        :return: string representation <value> <unit>
        :rtype: str
        """
        return f"{self.value:.2f} {self.unit}"
    
    def integrate_over_time(self, time_seconds: float) -> 'DataUnit':
        """
        Integrate the data rate over a given time in seconds to get total data volume.

        :time_seconds: time in seconds to integrate over
        :type time_seconds: float
        :return: DataUnit representing total data volume
        :rtype: DataUnit
        """
        if self.unit in [
            DataUnit.BYTES_PER_SECOND,
            DataUnit.KILOBYTES_PER_SECOND,
            DataUnit.MEGABYTES_PER_SECOND,
            DataUnit.GIGABYTES_PER_SECOND,
        ]:
            total_value = self.value * time_seconds
            volume_unit = self.unit.replace("/s", "")
            return DataUnit(total_value, volume_unit)
        elif self.unit in [
            DataUnit.BYTES_PER_MINUTE,
            DataUnit.KILOBYTES_PER_MINUTE,
            DataUnit.MEGABYTES_PER_MINUTE,
            DataUnit.GIGABYTES_PER_MINUTE,
        ]:
            total_value = self.value * (time_seconds / 60)
            volume_unit = self.unit.replace("/min", "")
            return DataUnit(total_value, volume_unit)
        elif self.unit in [
            DataUnit.BYTES_PER_HOUR,
            DataUnit.KILOBYTES_PER_HOUR,
            DataUnit.MEGABYTES_PER_HOUR,
            DataUnit.GIGABYTES_PER_HOUR,
        ]:
            total_value = self.value * (time_seconds / 3600)
            volume_unit = self.unit.replace("/h", "")
            return DataUnit(total_value, volume_unit)
        else:
            warnings.warn(
                "Integration can only be performed on data rates (per units of time)."
                " Returning original value.",
                UserWarning
            )
            return DataUnit(self.value, self.unit)

    def to(self, unit: str) -> 'DataUnit':
        """
        Convert the DataUnit to a different unit.
        This method supports conversion between all supported units.
        If converting between rates and volumes, an error is raised.

        :unit: target unit to convert to
        :type unit: str
        :return: new DataUnit instance in the target unit
        :rtype: DataUnit
        """
        unit = unit.lower()
        if not DataUnit.is_valid_unit(unit):
            raise ValueError(f"Invalid unit '{unit}'. Valid units are: {DataUnit.all_units()}")
        
        # Prevent conversion between rates and volumes
        if (
            len(unit.split('/')) == 2 and len(self.unit.split('/')) == 1
            or
            len(unit.split('/')) == 1 and len(self.unit.split('/')) == 2
        ):
            raise ValueError(
                "Cannot convert between data rates and data volumes."
                f" Current unit: '{self.unit}', target unit: '{unit}'"
            )

        # Conversion factors to convert FROM each unit TO bytes/s (or bytes for volume units)
        to_base_factors = {
            DataUnit.BYTES_PER_SECOND: 1,
            DataUnit.KILOBYTES_PER_SECOND: 1024,
            DataUnit.MEGABYTES_PER_SECOND: 1024 ** 2,
            DataUnit.GIGABYTES_PER_SECOND: 1024 ** 3,
            DataUnit.BYTES_PER_HOUR: 1 / 3600,
            DataUnit.KILOBYTES_PER_HOUR: 1024 / 3600,
            DataUnit.MEGABYTES_PER_HOUR: (1024 ** 2) / 3600,
            DataUnit.GIGABYTES_PER_HOUR: (1024 ** 3) / 3600,
            DataUnit.BYTES_PER_MINUTE: 1 / 60,
            DataUnit.KILOBYTES_PER_MINUTE: 1024 / 60,
            DataUnit.MEGABYTES_PER_MINUTE: (1024 ** 2) / 60,
            DataUnit.GIGABYTES_PER_MINUTE: (1024 ** 3) / 60,
            DataUnit.BYTES: 1,
            DataUnit.KILOBYTES: 1024,
            DataUnit.MEGABYTES: 1024 ** 2,
            DataUnit.GIGABYTES: 1024 ** 3,
        }

        # Convert current value to base unit (bytes/s or bytes)
        base_value = self.value * to_base_factors[self.unit]
        
        # Convert from base unit to target unit
        target_value = base_value / to_base_factors[unit]
        
        return DataUnit(target_value, unit)

    @staticmethod
    def all_units() -> list[str]:
        """
        Get a list of all valid units.
        
        :return: list of valid unit strings
        :rtype: list[str]
        """
        return [
            DataUnit.BYTES_PER_SECOND,
            DataUnit.KILOBYTES_PER_SECOND,
            DataUnit.MEGABYTES_PER_SECOND,
            DataUnit.GIGABYTES_PER_SECOND,
            DataUnit.BYTES_PER_HOUR,
            DataUnit.KILOBYTES_PER_HOUR,
            DataUnit.MEGABYTES_PER_HOUR,
            DataUnit.GIGABYTES_PER_HOUR,
            DataUnit.BYTES_PER_MINUTE,
            DataUnit.KILOBYTES_PER_MINUTE,
            DataUnit.MEGABYTES_PER_MINUTE,
            DataUnit.GIGABYTES_PER_MINUTE,
            DataUnit.BYTES,
            DataUnit.KILOBYTES,
            DataUnit.MEGABYTES,
            DataUnit.GIGABYTES,
        ]
    
    @staticmethod
    def is_valid_unit(unit: str) -> bool:
        """
        Check if a unit string is valid.

        :unit: unit string to check
        :return: True if valid, False otherwise
        :rtype: bool
        """
        return unit in DataUnit.all_units()


def data_rate(
    samples_per_second: Union[float, int],
    bytes_per_c_sample: Union[float, int],
    number_of_channels: int,
    number_of_antennas: int,
    unit: str = "bytes/s"
) -> DataUnit:
    """
    We define the sample rate F_s in samples/sec, N_b the bytes per complex
    sample, N_chann the number of channels, and N_ant the number of antennas.
    The data rate will be given by:
        Data Rate = F_s * N_b * N_chann * N_ant

    :samples_per_second: sample rate in samples per second
    :type samples_per_second: float or int
    :bytes_per_c_sample: bytes per complex sample
    :type bytes_per_c_sample: float or int
    :number_of_channels: number of channels
    :type number_of_channels: int
    :number_of_antennas: number of antennas
    :type number_of_antennas: int
    :unit: unit of the returned data rate (bytes/s, kb/s, mb/s, gb/s)
    :type unit: str
    :return: data rate in the specified unit
    :rtype: float
    """
    data_rate_bytes = DataUnit(
        samples_per_second * bytes_per_c_sample * number_of_channels * number_of_antennas,
        DataUnit.BYTES_PER_SECOND
    )
    return data_rate_bytes.to(unit)

def total_data_volume(
    data_rate: DataUnit,
    observation_time: float,
    unit: str = "bytes"
) -> DataUnit:
    """
    Calculate the total data volume.

    :data_rate: data rate in bytes per second
    :type data_rate: float
    :observation_time: observation time in seconds
    :type observation_time: float
    :unit: unit of the returned data volume (bytes, kb, mb, gb)
    :type unit: str
    :return: total data volume in the specified unit
    :rtype: float
    """
    return data_rate.integrate_over_time(observation_time).to(unit)
