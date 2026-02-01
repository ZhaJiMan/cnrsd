from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from io import BytesIO
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import ba2int
from numpy.typing import NDArray

__all__ = [
    "DIAMETER_CLASSES_100",
    "DIAMETER_CLASSES_200",
    "RSD",
    "VELOCITY_CLASSES_100",
    "VELOCITY_CLASSES_200",
    "RSDDecodeError",
    "rsds_to_dataframe",
]

VELOCITY_CLASSES_100 = np.array(
    [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
        (1.0, 1.2),
        (1.2, 1.4),
        (1.4, 1.6),
        (1.6, 1.8),
        (1.8, 2.0),
        (2.0, 2.4),
        (2.4, 2.8),
        (2.8, 3.2),
        (3.2, 3.6),
        (3.6, 4.0),
        (4.0, 4.8),
        (4.8, 5.6),
        (5.6, 6.4),
        (6.4, 7.2),
        (7.2, 8.0),
        (8.0, 9.6),
        (9.6, 11.2),
        (11.2, 12.8),
        (12.8, 14.4),
        (14.4, 16.0),
        (16.0, 19.2),
        (19.2, 22.4),
    ],
    dtype=np.float64,
)

DIAMETER_CLASSES_100 = np.array(
    [
        (0.0, 0.125),
        (0.125, 0.25),
        (0.25, 0.375),
        (0.375, 0.5),
        (0.5, 0.625),
        (0.625, 0.75),
        (0.75, 0.875),
        (0.875, 1.0),
        (1.0, 1.125),
        (1.125, 1.25),
        (1.25, 1.5),
        (1.5, 1.75),
        (1.75, 2.0),
        (2.0, 2.25),
        (2.25, 2.5),
        (2.5, 3.0),
        (3.0, 3.5),
        (3.5, 4.0),
        (4.0, 4.5),
        (4.5, 5.0),
        (5.0, 6.0),
        (6.0, 7.0),
        (7.0, 8.0),
        (8.0, 9.0),
        (9.0, 10.0),
        (10.0, 12.0),
        (12.0, 14.0),
        (14.0, 16.0),
        (16.0, 18.0),
        (18.0, 20.0),
        (20.0, 23.0),
        (23.0, 26.0),
    ],
    dtype=np.float64,
)

VELOCITY_CLASSES_200 = np.array(
    [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
        (1.0, 1.4),
        (1.4, 1.8),
        (1.8, 2.2),
        (2.2, 2.6),
        (2.6, 3.0),
        (3.0, 3.4),
        (3.4, 4.2),
        (4.2, 5.0),
        (5.0, 5.8),
        (5.8, 6.6),
        (6.6, 7.4),
        (7.4, 8.2),
        (8.2, 9.0),
        (9.0, 10.0),
        (10.0, 20.0),
    ],
    dtype=np.float64,
)

DIAMETER_CLASSES_200 = np.array(
    [
        (0.125, 0.25),
        (0.25, 0.375),
        (0.375, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
        (1.0, 1.25),
        (1.25, 1.5),
        (1.5, 1.75),
        (1.75, 2.0),
        (2.0, 2.5),
        (2.5, 3.0),
        (3.0, 3.5),
        (3.5, 4.0),
        (4.0, 4.5),
        (4.5, 5.0),
        (5.0, 5.5),
        (5.5, 6.0),
        (6.0, 6.5),
        (6.5, 7.0),
        (7.0, 7.5),
        (7.5, 8.0),
        (8.0, np.inf),
    ],
    dtype=np.float64,
)


def _decode_section4_size(section4: bitarray) -> int:
    return ba2int(section4[:24])


def _decode_wmo_id(section4: bitarray) -> str:
    block_number = ba2int(section4[32:39])
    station_number = ba2int(section4[39:49])
    return f"{block_number:02d}{station_number:03d}"


def _decode_station_name(section4: bitarray) -> str:
    return section4[49:209].tobytes().decode().rstrip("\x00")


def _decode_station_id(section4: bitarray) -> str:
    # 优先使用 station_name，其次使用 wmo_id
    return _decode_station_name(section4) or _decode_wmo_id(section4)


def _decode_reference_time(section4: bitarray) -> datetime:
    year = ba2int(section4[293:305])
    month = ba2int(section4[305:309])
    day = ba2int(section4[309:315])
    hour = ba2int(section4[315:320])
    minute = ba2int(section4[320:326])

    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _decode_value(value: int, factor: int, offset: int) -> float:
    return (value + offset) / 10**factor


def _decode_lonlat(section4: bitarray) -> tuple[float, float]:
    i0, i1, i2 = 326, 351, 377
    lat = _decode_value(ba2int(section4[i0:i1]), 5, -9000000)
    lon = _decode_value(ba2int(section4[i1:i2]), 5, -18000000)
    return lon, lat


def _decode_sensor_status(section4: bitarray) -> int:
    return ba2int(section4[377:380])


def _decode_device_type(section4: bitarray) -> int:
    return ba2int(section4[380:384])


@dataclass
class _RSDData:
    observation_times: list[float] = field(default_factory=list)
    rain_flags: list[bool] = field(default_factory=list)
    class_numbers: list[int] = field(default_factory=list)
    particle_numbers: list[int] = field(default_factory=list)

    def append(
        self, obs_time: float, rain_flag: bool, class_number: int, particle_number: int
    ) -> None:
        self.observation_times.append(obs_time)
        self.rain_flags.append(rain_flag)
        self.class_numbers.append(class_number)
        self.particle_numbers.append(particle_number)


def _decode_rsd_data(section4: bitarray, ref_time: datetime) -> _RSDData:
    """
    particle_number 的位数全 1 时即缺测，跳过缺测的记录。
    rep_factor=0 时视为无雨，bufr 里为了节省空间没有存计数。但为了方便后续处理，
    插入一条 rain_flag=False, class_number=1, particle_number=0 的记录表示无雨
    """
    # 这些量不从 bufr 中读取，而是使用常量
    # 如果 bufr 格式真的发生变化，在 archive 侧检查数值是否合理
    missing_value = 2**16 - 1
    time_increment = -5 * 60
    short_time_increment = 60
    rep_factor_7 = 5

    obs_time = ref_time.timestamp() + time_increment
    rep_factor_11 = section4[384]
    rsd_data = _RSDData()

    # 插入空值
    if rep_factor_11 == 0:
        for _ in range(rep_factor_7):
            obs_time += short_time_increment
            rsd_data.append(obs_time, False, 1, 0)
        return rsd_data

    # 跳过读取 rep_factor_7 和时间增量
    pos = 385 + 12 + 8 + 8
    for _ in range(rep_factor_7):
        obs_time += short_time_increment
        rep_factor_5 = ba2int(section4[pos : pos + 16])
        pos += 16

        # 插入空值
        if rep_factor_5 == 0:
            rsd_data.append(obs_time, False, 1, 0)
            continue

        for _ in range(rep_factor_5):
            # 跳过读取质控码
            i0, i1, i2, i3, i4 = pos, pos + 12, pos + 18, pos + 26, pos + 42  # noqa: F841
            class_number = ba2int(section4[i0:i1])
            particle_number = ba2int(section4[i3:i4])
            # 跳过缺测的计数
            if particle_number != missing_value:
                rsd_data.append(obs_time, True, class_number, particle_number)
            pos = i4

    return rsd_data


class RSDDecodeError(Exception):
    pass


def _to_datetime64_us(timestamps: Sequence[float]) -> NDArray[np.datetime64]:
    return (
        (np.array(timestamps, dtype=np.float64) * 1e9)
        .astype(np.int64)
        .astype("datetime64[us]")
    )


@dataclass(eq=False)
class RSD:
    station_id: str
    longitude: float
    latitude: float
    sensor_status: int
    device_type: int
    reference_time: datetime
    observation_times: NDArray[np.datetime64] = field(repr=False)
    rain_flags: NDArray[np.bool_] = field(repr=False)
    class_numbers: NDArray[np.int64] = field(repr=False)
    particle_numbers: NDArray[np.int64] = field(repr=False)

    def __post_init__(self) -> None:
        # 存在 device_type 跟 class_number 不匹配的情况
        if self.class_numbers.size > 0:
            if self.device_type == 0:
                velocity_classes = VELOCITY_CLASSES_100
                diameter_classes = DIAMETER_CLASSES_100
            else:
                velocity_classes = VELOCITY_CLASSES_200
                diameter_classes = DIAMETER_CLASSES_200
            num_classes = len(velocity_classes) * len(diameter_classes)
            if self.class_numbers.max() > num_classes:
                raise RSDDecodeError("class_number out of range")

    @classmethod
    def from_bytes(cls, data: bytes):
        header_size = 43
        section0_size = 8
        section1_size = 23
        section3_size = 9
        section5_size = 4
        trailer_size = 4  # noqa: F841

        with BytesIO(data) as f:
            f.seek(header_size)
            section0 = f.read(section0_size)
            if section0[:4] != b"BUFR":
                raise RSDDecodeError(
                    f"section0 的开头应该是 b'BUFR'，实际为 {section0[:4]}"
                )

            bufr_size = int.from_bytes(section0[4:7], byteorder="big")
            section4_size = (
                bufr_size
                - section0_size
                - section1_size
                - section3_size
                - section5_size
            )
            f.seek(f.tell() + section1_size + section3_size)
            section4 = bitarray(f.read(section4_size))

            section5 = f.read(section5_size)
            if section5 != b"7777":
                raise RSDDecodeError(f"section5 应该是 b'7777'，实际为 {section5}")

        station_id = _decode_station_id(section4)
        ref_time = _decode_reference_time(section4)
        lon, lat = _decode_lonlat(section4)
        sensor_status = _decode_sensor_status(section4)
        device_type = _decode_device_type(section4)
        rsd_data = _decode_rsd_data(section4, ref_time)

        return cls(
            station_id=station_id,
            longitude=lon,
            latitude=lat,
            sensor_status=sensor_status,
            device_type=device_type,
            reference_time=ref_time,
            observation_times=_to_datetime64_us(rsd_data.observation_times),
            rain_flags=np.array(rsd_data.rain_flags, dtype=np.bool_),
            class_numbers=np.array(rsd_data.class_numbers, dtype=np.int64),
            particle_numbers=np.array(rsd_data.particle_numbers, dtype=np.int64),
        )

    @classmethod
    def from_file(cls, filepath: str | PathLike[str]):
        # 小文件直接读入内存
        with open(filepath, mode="rb") as f:
            return cls.from_bytes(f.read())

    def to_dataframe(self) -> pd.DataFrame:
        return rsds_to_dataframe([self])


_DTYPE_MAP = {
    "station_id": "object",
    "longitude": "float64",
    "latitude": "float64",
    "time": "datetime64[us, UTC]",
    "sensor_status": "int64",
    "device_type": "int64",
    "rain_flag": "bool",
    "class_number": "int64",
    "particle_number": "int64",
    "velocity_min": "float64",
    "velocity_max": "float64",
    "diameter_min": "float64",
    "diameter_max": "float64",
}


@cache
def _make_empty_dataframe() -> pd.DataFrame:
    cols = pd.Index(list(_DTYPE_MAP.keys()))
    return pd.DataFrame(columns=cols).astype(_DTYPE_MAP)


def _pluck(iterable: Iterable[Any], name: str) -> list[Any]:
    return [getattr(obj, name) for obj in iterable]


def _make_class_product(
    velocity_classes: NDArray[np.float64], diameter_classes: NDArray[np.float64]
) -> NDArray[np.float64]:
    num_rows = len(velocity_classes)
    num_cols = len(diameter_classes)
    class_indices = np.arange(num_rows * num_cols)
    velocity_indices, diameter_indices = np.divmod(class_indices, num_cols)
    velocity_classes = velocity_classes[velocity_indices]
    diameter_classes = diameter_classes[diameter_indices]
    class_product = np.hstack([velocity_classes, diameter_classes])

    return class_product


@cache
def _get_class_table() -> NDArray[np.float64]:
    return np.vstack(
        [
            _make_class_product(VELOCITY_CLASSES_100, DIAMETER_CLASSES_100),
            _make_class_product(VELOCITY_CLASSES_200, DIAMETER_CLASSES_200),
        ]
    )


def _lookup_class_table(
    device_types: NDArray[np.int64], class_numbers: NDArray[np.int64]
) -> NDArray[np.float64]:
    class_table = _get_class_table()
    num_classes_100 = len(VELOCITY_CLASSES_100) * len(DIAMETER_CLASSES_100)
    class_indices = class_numbers - 1
    class_indices[device_types.astype(np.bool_)] += num_classes_100
    class_product = class_table[class_indices]

    return class_product


def rsds_to_dataframe(rsds: Sequence[RSD]) -> pd.DataFrame:
    """将多个 RSD 对象转换为 dataframe"""
    # 需要提前处理空列表，否则后面的 repeat 和 concatenate 会报错
    if len(rsds) == 0:
        return _make_empty_dataframe()
    repeats = np.array([len(rsd.observation_times) for rsd in rsds], dtype=np.int64)
    if (repeats == 0).all():
        return _make_empty_dataframe()

    # 省略 ref_time 字段
    data = {
        "station_id": np.repeat(
            np.array(_pluck(rsds, "station_id"), dtype=np.object_), repeats
        ),
        "longitude": np.repeat(
            np.array(_pluck(rsds, "longitude"), dtype=np.float64), repeats
        ),
        "latitude": np.repeat(
            np.array(_pluck(rsds, "latitude"), dtype=np.float64), repeats
        ),
        "time": pd.Series(
            np.concatenate(_pluck(rsds, "observation_times")),
            dtype="datetime64[us, UTC]",
        ),
        "sensor_status": np.repeat(
            np.array(_pluck(rsds, "sensor_status"), dtype=np.int64), repeats
        ),
        "device_type": np.repeat(
            np.array(_pluck(rsds, "device_type"), dtype=np.int64), repeats
        ),
        "rain_flag": np.concatenate(_pluck(rsds, "rain_flags")),
        "class_number": np.concatenate(_pluck(rsds, "class_numbers")),
        "particle_number": np.concatenate(_pluck(rsds, "particle_numbers")),
    }

    class_product = _lookup_class_table(data["device_type"], data["class_number"])
    data["velocity_min"] = class_product[:, 0]
    data["velocity_max"] = class_product[:, 1]
    data["diameter_min"] = class_product[:, 2]
    data["diameter_max"] = class_product[:, 3]
    df = pd.DataFrame(data).astype(_DTYPE_MAP)

    return df
