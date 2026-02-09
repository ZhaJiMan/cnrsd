from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from io import BytesIO
from os import PathLike
from typing import Any, Literal, NamedTuple, TypeAlias, TypedDict, cast, overload

import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import ba2int
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "RSD",
    "RSD_GRID_100",
    "RSD_GRID_200",
    "BinAxis",
    "ClassParams",
    "DeviceType",
    "RSDDict",
    "RSDError",
    "RSDGrid",
    "SensorStatus",
    "get_rsd_grid",
    "lookup_class_params",
    "resample_rsd_dataframe",
    "rsds_to_dataframe",
    "rsds_to_dict",
]

# 根据 BUFR 表格已知的常量
_MISSING_VALUE = 2**16 - 1
_TIME_INCREMENT = -5
_SHORT_TIME_INCREMENT = 1
_REP_FACTOR_7 = 5

# section4 的大小不是固定的
_HEADER_SIZE = 43
_SECTION0_SIZE = 8
_SECTION1_SIZE = 23
_SECTION3_SIZE = 9
_SECTION5_SIZE = 4
_TRAILER_SIZE = 4


@dataclass
class BinAxis:
    edges: NDArray[np.float64] = field(repr=False)
    num_bins: int = field(init=False)
    lower_bounds: NDArray[np.float64] = field(init=False, repr=False)
    upper_bounds: NDArray[np.float64] = field(init=False, repr=False)
    centers: NDArray[np.float64] = field(init=False, repr=False)
    widths: NDArray[np.float64] = field(init=False, repr=False)

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    @classmethod
    def from_edges(cls, edges: Sequence[float] | NDArray[np.floating]):
        return cls(np.asarray(edges, dtype=np.float64))

    def __post_init__(self) -> None:
        assert self.edges.ndim == 1 and len(self.edges) >= 2
        self.num_bins = len(self.edges) - 1
        self.lower_bounds = self.edges[:-1]
        self.upper_bounds = self.edges[1:]
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.widths = self.edges[1:] - self.edges[:-1]


@dataclass
class RSDGrid:
    velocity: BinAxis
    diameter: BinAxis
    shape: tuple[int, int] = field(init=False, repr=False)
    num_classes: int = field(init=False, repr=False)

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        self.shape = (self.velocity.num_bins, self.diameter.num_bins)
        self.num_classes = self.shape[0] * self.shape[1]


RSD_GRID_100 = RSDGrid(
    velocity=BinAxis.from_edges(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.2,
            1.4,
            1.6,
            1.8,
            2.0,
            2.4,
            2.8,
            3.2,
            3.6,
            4.0,
            4.8,
            5.6,
            6.4,
            7.2,
            8.0,
            9.6,
            11.2,
            12.8,
            14.4,
            16.0,
            19.2,
            22.4,
        ]
    ),
    diameter=BinAxis.from_edges(
        [
            0.0,
            0.125,
            0.25,
            0.375,
            0.5,
            0.625,
            0.75,
            0.875,
            1.0,
            1.125,
            1.25,
            1.5,
            1.75,
            2.0,
            2.25,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            23.0,
            26.0,
        ]
    ),
)

RSD_GRID_200 = RSDGrid(
    velocity=BinAxis.from_edges(
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.4,
            1.8,
            2.2,
            2.6,
            3.0,
            3.4,
            4.2,
            5.0,
            5.8,
            6.6,
            7.4,
            8.2,
            9.0,
            10.0,
            20.0,
        ]
    ),
    diameter=BinAxis.from_edges(
        [
            0.125,
            0.25,
            0.375,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            9.0,  # 用 9.0 替代 inf
        ]
    ),
)


SensorStatus: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7]
DeviceType: TypeAlias = Literal[0, 1]


def get_rsd_grid(device_type: DeviceType) -> RSDGrid:
    match device_type:
        case 0:
            return RSD_GRID_100
        case 1:
            return RSD_GRID_200
        case _:
            raise RSDError(f"device_type 的值应该是 0 或 1，实际是 {device_type}")


class RSDError(Exception):
    pass


def _decode_wmo_station_id(section4: bitarray) -> str:
    block_number = ba2int(section4[32:39])
    station_number = ba2int(section4[39:49])
    return f"{block_number:02d}{station_number:03d}"


def _decode_local_station_id(section4: bitarray) -> str:
    data = section4[49:209].tobytes()
    try:
        string = data.decode("ascii")
    except UnicodeDecodeError as e:
        raise RSDError("本地测站标识的值应该是 ASCII 编码") from e

    return string.rstrip("\x00")


def _decode_station_id(section4: bitarray) -> str:
    # 非 WMO 区站使用本地测站标识
    return _decode_local_station_id(section4) or _decode_wmo_station_id(section4)


def _decode_ref_time(section4: bitarray) -> datetime:
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


def _decode_sensor_status(section4: bitarray) -> SensorStatus:
    sensor_status = ba2int(section4[377:380])
    if sensor_status not in {0, 1, 2, 3, 4, 5, 6, 7}:
        raise RSDError(
            f"雨滴谱传感器标识的值应该在 0 到 7 范围内，实际是 {sensor_status}"
        )

    return cast(SensorStatus, sensor_status)


def _decode_device_type(section4: bitarray) -> DeviceType:
    device_type = ba2int(section4[380:384])
    if device_type not in {0, 1}:
        raise RSDError(f"雨滴谱设备类型的值应该是 0 或 1，实际是 {device_type}")

    return cast(DeviceType, device_type)


def _decode_time_increment(section4: bitarray) -> float:
    time_increment = _decode_value(ba2int(section4[385:397]), 0, -2048)
    if not math.isclose(time_increment, _TIME_INCREMENT):
        raise RSDError(f"时间增量的值应该是 {_TIME_INCREMENT}，实际是 {time_increment}")

    return time_increment


def _decode_short_time_increment(section4: bitarray) -> float:
    short_time_increment = _decode_value(ba2int(section4[397:405]), 0, -128)
    if not math.isclose(short_time_increment, _SHORT_TIME_INCREMENT):
        raise RSDError(
            f"短时间增量的值应该是 {_SHORT_TIME_INCREMENT}，实际是 {short_time_increment}"
        )

    return short_time_increment


def _decode_rep_factor_11(section4: bitarray) -> Literal[0, 1]:
    return cast(Literal[0, 1], section4[384])


def _decode_rep_factor_7(section4: bitarray) -> Literal[5]:
    rep_factor_7 = ba2int(section4[405:413])
    if rep_factor_7 != _REP_FACTOR_7:
        raise RSDError(
            f"延迟描述符重复因子的值应该是 {_REP_FACTOR_7}，实际是 {rep_factor_7}"
        )

    return cast(Literal[5], rep_factor_7)


@dataclass
class _RSDBody:
    times: list[float] = field(default_factory=list)
    rain_flags: list[bool] = field(default_factory=list)
    class_numbers: list[int] = field(default_factory=list)
    particle_numbers: list[int] = field(default_factory=list)

    def append(
        self, time: float, rain_flag: bool, class_number: int, particle_number: int
    ) -> None:
        self.times.append(time)
        self.rain_flags.append(rain_flag)
        self.class_numbers.append(class_number)
        self.particle_numbers.append(particle_number)


def _decode_rsd_body(section4: bitarray, ref_time: datetime) -> _RSDBody:
    """
    particle_number 的位数全 1 时即缺测，跳过缺测的记录。
    rep_factor=0 时视为无雨，bufr 里为了节省空间没有存计数。但为了方便后续处理，
    插入一条 rain_flag=False, class_number=1, particle_number=0 的记录表示无雨
    """
    # time 的单位是秒
    time = ref_time.timestamp() + _TIME_INCREMENT * 60

    # 插入空值
    rsd_body = _RSDBody()
    rep_factor_11 = _decode_rep_factor_11(section4)
    if rep_factor_11 == 0:
        for _ in range(_REP_FACTOR_7):
            time += _SHORT_TIME_INCREMENT * 60
            rsd_body.append(time, False, 1, 0)
        return rsd_body

    # 通过解码检查常量的值是否符合预期，以防 BUFR 格式发生变化
    time_increment = _decode_time_increment(section4)  # noqa: F841
    short_time_increment = _decode_short_time_increment(section4)  # noqa: F841
    rep_factor_7 = _decode_rep_factor_7(section4)  # noqa: F841

    pos = 413
    for _ in range(_REP_FACTOR_7):
        time += _SHORT_TIME_INCREMENT * 60
        rep_factor_5 = ba2int(section4[pos : pos + 16])
        pos += 16

        # 插入空值
        if rep_factor_5 == 0:
            rsd_body.append(time, False, 1, 0)
            continue

        for _ in range(rep_factor_5):
            # 因为质控码始终是 0，所以跳过读取
            i0, i1, i2, i3, i4 = pos, pos + 12, pos + 18, pos + 26, pos + 42  # noqa: F841
            class_number = ba2int(section4[i0:i1])
            particle_number = ba2int(section4[i3:i4])
            if particle_number != _MISSING_VALUE:  # 跳过缺测
                rsd_body.append(time, True, class_number, particle_number)
            pos = i4

    return rsd_body


def _to_datetime64_us(times: Sequence[float]) -> NDArray[np.datetime64]:
    return (
        (np.array(times, dtype=np.float64) * 1e6)
        .astype(np.int64)
        .astype("datetime64[us]")
    )


@dataclass
class RSD:
    station_id: str
    longitude: float
    latitude: float
    sensor_status: SensorStatus
    device_type: DeviceType
    reference_time: datetime
    times: NDArray[np.datetime64] = field(repr=False)
    rain_flags: NDArray[np.bool_] = field(repr=False)
    class_numbers: NDArray[np.int64] = field(repr=False)
    particle_numbers: NDArray[np.int64] = field(repr=False)
    num_records: int = field(init=False)
    grid: RSDGrid = field(init=False, repr=False)

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        self.num_records = len(self.times)
        self.grid = get_rsd_grid(self.device_type)

        # 存在 device_type 跟 class_number 不匹配的情况
        if self.num_records > 0:
            rsd_grid = get_rsd_grid(self.device_type)
            max_class_number = self.class_numbers.max()
            if max_class_number > rsd_grid.num_classes:
                raise RSDError(
                    f"class_number 的最大值 {max_class_number} 超过了"
                    f"device_type={self.device_type} 允许的上限 {rsd_grid.num_classes}"
                )

    @classmethod
    def from_bytes(cls, data: bytes):
        with BytesIO(data) as f:
            f.seek(_HEADER_SIZE)
            section0 = f.read(_SECTION0_SIZE)
            if section0[:4] != b"BUFR":
                raise RSDError(f"section0 的开头应该是 b'BUFR'，实际是 {section0[:4]}")

            bufr_size = int.from_bytes(section0[4:7], byteorder="big")
            section4_size = (
                bufr_size
                - _SECTION0_SIZE
                - _SECTION1_SIZE
                - _SECTION3_SIZE
                - _SECTION5_SIZE
            )
            f.seek(f.tell() + _SECTION1_SIZE + _SECTION3_SIZE)
            section4 = f.read(section4_size)

            section5 = f.read(_SECTION5_SIZE)
            if section5 != b"7777":
                raise RSDError(f"section5 的值应该是 b'7777'，实际是 {section5}")

        section4 = bitarray(section4)
        station_id = _decode_station_id(section4)
        ref_time = _decode_ref_time(section4)
        lon, lat = _decode_lonlat(section4)
        sensor_status = _decode_sensor_status(section4)
        device_type = _decode_device_type(section4)
        rsd_body = _decode_rsd_body(section4, ref_time)

        return cls(
            station_id=station_id,
            longitude=lon,
            latitude=lat,
            sensor_status=sensor_status,
            device_type=device_type,
            reference_time=ref_time,
            times=_to_datetime64_us(rsd_body.times),
            rain_flags=np.array(rsd_body.rain_flags, dtype=np.bool_),
            class_numbers=np.array(rsd_body.class_numbers, dtype=np.int64),
            particle_numbers=np.array(rsd_body.particle_numbers, dtype=np.int64),
        )

    @classmethod
    def from_file(cls, filepath: str | PathLike[str]):
        # 小文件直接读入内存
        with open(filepath, mode="rb") as f:
            return cls.from_bytes(f.read())

    @overload
    def to_dict(self, include_class_params: Literal[False]) -> RSDDict: ...

    @overload
    def to_dict(self, include_class_params: Literal[True] = True) -> FullRSDDict: ...

    def to_dict(self, include_class_params: bool = True) -> RSDDict | FullRSDDict:
        return rsds_to_dict([self], include_class_params)

    def to_dataframe(self) -> pd.DataFrame:
        return rsds_to_dataframe([self])


def _vstack_bin_params(bin_axis: BinAxis) -> NDArray[np.float64]:
    return np.column_stack(
        (
            bin_axis.lower_bounds,
            bin_axis.upper_bounds,
            bin_axis.centers,
            bin_axis.widths,
        )
    )


def _make_class_params(rsd_grid: RSDGrid) -> NDArray[np.float64]:
    class_indices = np.arange(rsd_grid.num_classes)
    velocity_indices, diameter_indices = np.divmod(class_indices, rsd_grid.shape[1])
    velocity_params = _vstack_bin_params(rsd_grid.velocity)[velocity_indices, :]
    diameter_params = _vstack_bin_params(rsd_grid.diameter)[diameter_indices, :]
    class_params = np.hstack([velocity_params, diameter_params])

    return class_params


@cache
def _get_class_table() -> NDArray[np.float64]:
    return np.vstack(
        (_make_class_params(RSD_GRID_100), _make_class_params(RSD_GRID_200))
    )


class ClassParams(NamedTuple):
    velocity_lower_bounds: NDArray[np.float64]
    velocity_upper_bounds: NDArray[np.float64]
    velocity_centers: NDArray[np.float64]
    velocity_widths: NDArray[np.float64]
    diameter_lower_bounds: NDArray[np.float64]
    diameter_upper_bounds: NDArray[np.float64]
    diameter_centers: NDArray[np.float64]
    diameter_widths: NDArray[np.float64]

    __hash__ = None  # pyright: ignore[reportAssignmentType]


def lookup_class_params(
    device_types: ArrayLike, class_numbers: ArrayLike
) -> ClassParams:
    # 允许输入是标量
    device_types = np.asarray(device_types)
    if not ((device_types == 0) | (device_types == 1)).all():
        raise RSDError("device_types 的元素的值只能是 0 或 1")

    # np.where 会进行广播
    class_indices = np.asarray(class_numbers, dtype=np.intp) - 1
    class_indices = np.where(
        device_types.astype(np.bool_),
        class_indices + RSD_GRID_100.num_classes,
        class_indices,
    )

    class_table = _get_class_table()

    try:
        # advanced indexing 返回 copy
        class_params = class_table[class_indices, :]
    except IndexError as e:
        raise RSDError("class_numbers 有元素的值超过了 device_types 允许的上限") from e

    # 使用 ellipsis 至少会返回零维数组
    args = (class_params[..., i] for i in range(class_params.shape[-1]))

    return ClassParams(*args)


def _pluck(iterable: Iterable[object], name: str) -> list[Any]:
    return [getattr(obj, name) for obj in iterable]


class RSDDict(TypedDict):
    station_id: NDArray[np.str_]
    longitude: NDArray[np.float64]
    latitude: NDArray[np.float64]
    time: NDArray[np.datetime64]
    sensor_status: NDArray[np.int64]
    device_type: NDArray[np.int64]
    rain_flag: NDArray[np.bool_]
    class_number: NDArray[np.int64]
    particle_number: NDArray[np.int64]


class FullRSDDict(RSDDict):
    velocity_center: NDArray[np.float64]
    velocity_width: NDArray[np.float64]
    diameter_center: NDArray[np.float64]
    diameter_width: NDArray[np.float64]


@overload
def rsds_to_dict(
    rsds: Sequence[RSD], include_class_params: Literal[False]
) -> RSDDict: ...


@overload
def rsds_to_dict(
    rsds: Sequence[RSD], include_class_params: Literal[True] = True
) -> FullRSDDict: ...


def rsds_to_dict(
    rsds: Sequence[RSD], include_class_params: bool = True
) -> RSDDict | FullRSDDict:
    # 需要提前处理空列表，否则后面的 repeat 和 concatenate 会报错
    repeats = _pluck(rsds, "num_records")
    if sum(repeats) == 0:
        return {
            "station_id": np.array([], dtype=np.str_),
            "longitude": np.array([], dtype=np.float64),
            "latitude": np.array([], dtype=np.float64),
            "time": np.array([], dtype=np.datetime64),
            "sensor_status": np.array([], dtype=np.int64),
            "device_type": np.array([], dtype=np.int64),
            "rain_flag": np.array([], dtype=np.bool_),
            "class_number": np.array([], dtype=np.int64),
            "particle_number": np.array([], dtype=np.int64),
            "velocity_center": np.array([], dtype=np.float64),
            "velocity_width": np.array([], dtype=np.float64),
            "diameter_center": np.array([], dtype=np.float64),
            "diameter_width": np.array([], dtype=np.float64),
        }

    # 貌似 pluck 比 attrgetter + zip 要快一点
    repeats = np.array(repeats, dtype=np.int64)
    data = {
        "station_id": np.repeat(
            np.array(_pluck(rsds, "station_id"), dtype=np.str_), repeats
        ),
        "longitude": np.repeat(
            np.array(_pluck(rsds, "longitude"), dtype=np.float64), repeats
        ),
        "latitude": np.repeat(
            np.array(_pluck(rsds, "latitude"), dtype=np.float64), repeats
        ),
        "time": np.concatenate(_pluck(rsds, "times")),
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

    if include_class_params:
        class_params = lookup_class_params(data["device_type"], data["class_number"])
        data["velocity_center"] = class_params.velocity_centers
        data["velocity_width"] = class_params.velocity_widths
        data["diameter_center"] = class_params.diameter_centers
        data["diameter_width"] = class_params.diameter_widths

    return cast(RSDDict | FullRSDDict, data)


def rsds_to_dataframe(rsds: Sequence[RSD]) -> pd.DataFrame:
    """将多个 RSD 对象转换为 dataframe"""
    return pd.DataFrame(rsds_to_dict(rsds))


def resample_rsd_dataframe(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    # 普通列加上时间 grouper 会自动丢弃空时间窗口
    grouper = pd.Grouper(key="time", freq=freq, closed="right", label="right")  # pyright: ignore[reportCallIssue]
    window_flags = df.groupby(["station_id", grouper])["rain_flag"].transform("any")
    # 提前过滤时间窗口里占位的无雨行
    df = cast(pd.DataFrame, df[~window_flags | df["rain_flag"]])

    agg_map = {col: "first" for col in df.columns if col}
    for col in ["station_id", "time", "class_number"]:
        del agg_map[col]
    agg_map["particle_number"] = "sum"

    agg_df = (
        df.groupby(["station_id", grouper, "class_number"], as_index=False)
        .agg(agg_map)[df.columns]
        .reset_index(drop=True)  # pyright: ignore[reportAttributeAccessIssue]
    )

    return cast(pd.DataFrame, agg_df)
