from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from io import BytesIO
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias, TypedDict, cast

import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int
from numpy.typing import ArrayLike, DTypeLike, NDArray

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

__version__ = "0.1.2"

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
    "build_rsd_dataarray",
    "get_bin_edges",
    "get_rsd_grid",
    "gunn_kinzer_velocity",
    "lookup_class_params",
    "mass_weighted_diameter",
    "read_file",
    "resample_rsd_dataframe",
    "rsds_to_dataframe",
    "rsds_to_dict",
]

# 根据 BUFR 表格已知的常量
_MISSING_VALUE = 2**16 - 1
_TIME_INCREMENT = -5.0
_SHORT_TIME_INCREMENT = 1.0
_REP_FACTOR_7 = 5

# section4 的大小不是固定的
_HEADER_SIZE = 43
_SECTION0_SIZE = 8
_SECTION1_SIZE = 23
_SECTION3_SIZE = 9
_SECTION5_SIZE = 4
_TRAILER_SIZE = 4

# 站名只允许出现数字和英文字母
_STATION_ID_PATTERN = re.compile(r"[0-9a-zA-Z]*")


@dataclass
class BinAxis:
    """表示一维分箱的轴

    Attributes
    ----------
    edges : (n + 1,) ndarray
        分箱边缘

    num_bins : int
        分箱数量，等于 len(edges) - 1

    lower_bounds : (n,) ndarray
        每个分箱的左边缘

    upper_bounds : (n,) ndarray
        每个分箱的右边缘

    centers : (n,) ndarray
        每个分箱的中心

    widths : (n,) ndarray
        每个分箱的宽度
    """

    edges: NDArray[np.float64] = field(repr=False)
    num_bins: int = field(init=False)
    lower_bounds: NDArray[np.float64] = field(init=False, repr=False)
    upper_bounds: NDArray[np.float64] = field(init=False, repr=False)
    centers: NDArray[np.float64] = field(init=False, repr=False)
    widths: NDArray[np.float64] = field(init=False, repr=False)

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        if self.edges.ndim != 1 or len(self.edges) < 2:
            raise ValueError("edges 必须是长度至少为 2 的一维数组")
        if not np.all(self.edges[:-1] <= self.edges[1:]):
            raise ValueError("edges 必须单调递增")

        self.num_bins = len(self.edges) - 1
        self.lower_bounds = self.edges[:-1]
        self.upper_bounds = self.edges[1:]
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.widths = self.edges[1:] - self.edges[:-1]

    @classmethod
    def from_edges(cls, edges: Sequence[float] | NDArray[np.floating]):
        """用分箱边缘构造 BinAxis 对象"""
        return cls(np.asarray(edges, dtype=np.float64))


@dataclass
class RSDGrid:
    """雨滴谱的二维分箱网格

    Attributes
    ----------
    velocity : BinAxis
        速度分箱轴

    diameter : BinAxis
        直径分箱轴

    shape : tuple[int, int]
        网格形状，等于 (velocity.num_bins, diameter.num_bins)。

    num_classes : int
        网格含有的分级数量
    """

    velocity: BinAxis
    diameter: BinAxis
    shape: tuple[int, int] = field(init=False, repr=False)
    num_classes: int = field(init=False, repr=False)

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __post_init__(self) -> None:
        self.shape = (self.velocity.num_bins, self.diameter.num_bins)
        self.num_classes = self.shape[0] * self.shape[1]


# 100 型雨滴谱仪
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

# 200 型雨滴谱仪
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


def get_rsd_grid(device_type: int) -> RSDGrid:
    """根据设备类型获取雨滴谱的分箱网格"""
    match device_type:
        case 0:
            return RSD_GRID_100
        case 1:
            return RSD_GRID_200
        case _:
            raise ValueError(f"{device_type=} 应该是 0 或 1")


class RSDError(Exception):
    pass


def _decode_wmo_station_id(section4: bitarray) -> str:
    # 非 WMO 站点存在 block_number 为零的情况
    block_number = ba2int(section4[32:39])
    station_number = ba2int(section4[39:49])
    return f"{block_number:02d}{station_number:03d}"


def _decode_local_station_id(section4: bitarray) -> str:
    # 假设 WMO 站点的 local_station_id 全填充 \x00
    # 非 WMO 站点的 local_station_id 由字母、数字和 \x00 组成
    data = section4[49:209].tobytes()
    try:
        local_station_id = data.decode("ascii").rstrip("\x00")
    except UnicodeDecodeError as e:
        raise RSDError(f"{data=} 包含非 ASCII 字符") from e

    # 存在含有 \x03 的情况，这里不予接受
    if _STATION_ID_PATTERN.fullmatch(local_station_id) is None:
        raise RSDError(f"{local_station_id=} 含有数字和英文字母以外的字符")

    return local_station_id


def _decode_station_id(section4: bitarray) -> str:
    # 优先选择非空的 local_station_id
    return _decode_local_station_id(section4) or _decode_wmo_station_id(section4)


def _decode_ref_time(section4: bitarray) -> datetime:
    # TODO: 存在时间是 1718-06-28 的情况
    year = ba2int(section4[293:305])
    month = ba2int(section4[305:309])
    day = ba2int(section4[309:315])
    hour = ba2int(section4[315:320])
    minute = ba2int(section4[320:326])

    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _decode_value(value: int, factor: int, offset: int) -> float:
    return (value + offset) / 10**factor


def _decode_lonlat(section4: bitarray) -> tuple[float, float]:
    # TODO: 存在经纬度超出中国范围的情况
    i0, i1, i2 = 326, 351, 377
    lat = _decode_value(ba2int(section4[i0:i1]), 5, -9000000)
    lon = _decode_value(ba2int(section4[i1:i2]), 5, -18000000)

    return lon, lat


def _decode_sensor_status(section4: bitarray) -> SensorStatus:
    sensor_status = ba2int(section4[377:380])
    if sensor_status not in {0, 1, 2, 3, 4, 5, 6, 7}:
        raise RSDError(f"{sensor_status=} 应该是 0 到 7")

    return cast(SensorStatus, sensor_status)


def _decode_device_type(section4: bitarray) -> DeviceType:
    # 存在 device_type = 7 的情况
    device_type = ba2int(section4[380:384])
    if device_type not in {0, 1}:
        raise RSDError(f"{device_type=} 应该是 0 或 1")

    return cast(DeviceType, device_type)


def _decode_time_increment(section4: bitarray) -> float:
    time_increment = _decode_value(ba2int(section4[385:397]), 0, -2048)
    if time_increment != _TIME_INCREMENT:  # 没有浮点数比较问题
        raise RSDError(f"{time_increment=} 应该是 {_TIME_INCREMENT}")

    return time_increment


def _decode_short_time_increment(section4: bitarray) -> float:
    short_time_increment = _decode_value(ba2int(section4[397:405]), 0, -128)
    if short_time_increment != _SHORT_TIME_INCREMENT:
        raise RSDError(f"{short_time_increment=} 应该是 {_SHORT_TIME_INCREMENT}")

    return short_time_increment


def _decode_rep_factor_11(section4: bitarray) -> Literal[0, 1]:
    return cast(Literal[0, 1], section4[384])


def _decode_rep_factor_7(section4: bitarray) -> int:
    rep_factor_7 = ba2int(section4[405:413])
    if rep_factor_7 != _REP_FACTOR_7:
        raise RSDError(f"{rep_factor_7=} 应该是 {_REP_FACTOR_7}")

    return rep_factor_7


@dataclass
class _RSDBody:
    times: list[float] = field(default_factory=list)
    rain_flags: list[bool] = field(default_factory=list)
    class_numbers: list[int] = field(default_factory=list)
    particle_numbers: list[int] = field(default_factory=list)

    def append_record(
        self, time: float, class_number: int, particle_number: int
    ) -> None:
        self.times.append(time)
        self.rain_flags.append(True)
        self.class_numbers.append(class_number)
        self.particle_numbers.append(particle_number)

    def append_placeholder(self, time: float) -> None:
        self.times.append(time)
        self.rain_flags.append(False)
        self.class_numbers.append(1)
        self.particle_numbers.append(0)


def _decode_rsd_body(section4: bitarray, ref_time: datetime) -> _RSDBody:
    # 时间单位是秒
    time = ref_time.timestamp() + _TIME_INCREMENT * 60

    # rep_factor_11 为 0 表示 5 个时刻均无雨，插入 5 行占位行
    # 为了插入占位行，需要提前知道 rep_factor_7 和时间增量的值
    rsd_body = _RSDBody()
    rep_factor_11 = _decode_rep_factor_11(section4)
    if rep_factor_11 == 0:
        for _ in range(_REP_FACTOR_7):
            time += _SHORT_TIME_INCREMENT * 60
            rsd_body.append_placeholder(time)
        return rsd_body

    # 通过调用函数间接检查硬编码的常量是否跟 BUFR 文件一致
    time_increment = _decode_time_increment(section4)  # noqa: F841
    short_time_increment = _decode_short_time_increment(section4)  # noqa: F841
    rep_factor_7 = _decode_rep_factor_7(section4)  # noqa: F841

    pos = 413
    for _ in range(_REP_FACTOR_7):
        time += _SHORT_TIME_INCREMENT * 60
        rep_factor_5 = ba2int(section4[pos : pos + 16])
        pos += 16

        # rep_factor_5 为零表示对应时刻无雨，插入占位行
        if rep_factor_5 == 0:
            rsd_body.append_placeholder(time)
            continue

        for _ in range(rep_factor_5):
            i0, i1, i2, i3, i4 = pos, pos + 12, pos + 18, pos + 26, pos + 42  # noqa: F841
            class_number = ba2int(section4[i0:i1])
            # qc_significance = ba2int(section4[i1:i2])  # =62
            # qc_code = ba2int(section4[i2:i3])  # =0
            particle_number = ba2int(section4[i3:i4])
            # TODO: 可能存在部分分级的粒子数缺测的情况吗？
            if particle_number != _MISSING_VALUE:
                rsd_body.append_record(time, class_number, particle_number)
            pos = i4

    return rsd_body


def _to_datetime64_us(seconds: Sequence[float]) -> NDArray[np.datetime64]:
    return (
        (np.asarray(seconds, dtype=np.float64) * 1e6)
        .astype(np.int64)
        .astype("datetime64[us]")
    )


@dataclass
class RSD:
    """雨滴谱数据

    BUFR 文件从参考时间开始往前，每分钟观测一个雨滴谱，共观测 5 个雨滴谱。`RSD` 类将每个时刻、
    雨滴谱每个分级的粒子数以长表的形式保存。

    为了节省内存，只保存粒子数非零的行，并设置 `rain_flag=True`；如果某个时刻的所有分级的粒子数都是零，
    为了避免直接损失这个时刻，插入占位行（`rain_flag=False`, `class_number=1`, `particle_number=0`）。

    Attributes
    ----------
    station_id : str
        测站标识。优先使用本地测站标识，其次使用 WMO 站号。

    longitude : float
        站点经度

    latitude : float
        站点纬度

    sensor_status : {0, 1, 2, 3, 4, 5, 6, 7}
        传感器标识

        - 0：无观测任务
        - 1：自动观测
        - 2：人工观测
        - 3：加盖期间
        - 4：仪器故障期间
        - 5：仪器维护期间
        - 6：日落后日出前无数据
        - 7：缺测

    device_type : {0, 1}
        设备类型

        - 0：雨滴谱设备类型为 100，输出 32 级粒子大小与 32 级粒子速度
        - 1：雨滴谱设备类型为 200，输出 22 级粒子大小与 22 级粒子速度

    reference_time : datetime
        参考时间。含 UTC 时区信息

    times : (n,) ndarray
        每行对应的 UTC 时间。但不含时区信息

    rain_flags : (n,) ndarray
        `True` 表示该行所属的时刻有雨（存在至少一个分级的粒子数非零），
        `False` 表示该行所属的时刻无雨（所有分级的粒子数都是零）。

    class_numbers : (n,) ndarray
        每行对应的分级编号

    particle_numbers : (n,) ndarray
        每行对应的粒子数

    num_records : int
        长表行（记录）数

    grid : RSDGrid
        `device_type` 对应的雨滴谱分箱网格
    """

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

        # TODO: 检查 class_numbers 和 particle_numbers 的值是否合法
        # 存在 device_type 跟 class_number 不匹配的情况
        if self.num_records > 0:
            max_class_number = self.class_numbers.max()
            if max_class_number > self.grid.num_classes:
                raise RSDError(
                    f"class_numbers 的最大值 {max_class_number} 超过了"
                    f"device_type={self.device_type} 允许的上限 {self.grid.num_classes}"
                )

    @classmethod
    def from_bytes(cls, data: bytes):
        """从 bytes 构造 `RSD` 对象"""
        with BytesIO(data) as f:
            f.seek(_HEADER_SIZE)
            section0 = f.read(_SECTION0_SIZE)
            if section0[:4] != b"BUFR":
                raise RSDError(f"section0[:4]={section0[:4]} 应该是 b'BUFR'")

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
                raise RSDError(f"{section5=} 应该是 b'7777'")

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
        """从文件构造 `RSD` 对象"""
        # 小文件直接读入内存
        with open(filepath, mode="rb") as f:
            return cls.from_bytes(f.read())

    def to_dict(self) -> RSDDict:
        """将 `RSD` 对象转换成列式字典

        标量属性会广播到与行数相同的长度，行数为零时得到每列都是空数组的字典。

        会根据 `class_numbers` 追加对应的速度和直径分箱的中心和宽度的列，方便后续过滤和统计。

        结果可以传给 pandas 或 polars 以构造 `DataFrame` 对象。
        """
        return rsds_to_dict([self])

    def to_dataframe(self) -> pd.DataFrame:
        """将 `RSD` 对象转换成 pandas 的 `DataFrame` 对象"""
        return rsds_to_dataframe([self])

    def to_dataarray(self) -> xr.DataArray:
        """将 `RSD` 对象转换成多维的 xarray 的 `DataArray` 对象

        维度是 `(time, velocity_center, diameter_center)`，元素值是粒子数，
        `velocity_width` 和 `diameter_width` 是辅助坐标，标量属性保存到 `attrs` 中。
        不再含有 rain_flag 属性，直接用全零的粒子数表示。

        如果需要保存成 netCDF4 文件，需要用户自行处理 `attrs`，例如将 `reference_time` 属性
        从 datetime 类型转换成字符串。
        """
        da = build_rsd_dataarray(
            device_type=self.device_type,
            class_numbers=self.class_numbers,
            particle_numbers=self.particle_numbers,
            times=self.times,
        )

        da.attrs["station_id"] = self.station_id
        da.attrs["longitude"] = self.longitude
        da.attrs["latitude"] = self.latitude
        da.attrs["sensor_status"] = self.sensor_status
        da.attrs["device_type"] = self.device_type
        da.attrs["reference_time"] = self.reference_time

        return da


read_file = RSD.from_file


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
    """根据设备类型和分级编号查找对应的速度和直径的分箱参数（中心、宽度和上下界）"""
    device_types = np.atleast_1d(np.asarray(device_types))
    class_numbers = np.atleast_1d(np.asarray(class_numbers))
    if device_types.shape != class_numbers.shape:
        raise ValueError("device_types 和 class_numbers 的形状必须相同")

    # 存在 device_type = 7 的文件
    if not ((device_types == 0) | (device_types == 1)).all():
        raise ValueError("device_types 的元素的值只能是 0 或 1")
    if not np.issubdtype(class_numbers.dtype, np.integer):
        raise TypeError("class_numbers 必须是整数类型")
    if (class_numbers < 1).any():
        raise ValueError("class_numbers 的元素的值必须 >= 1")

    mask_200 = device_types.astype(np.bool_)
    max_numbers = np.where(mask_200, RSD_GRID_200.num_classes, RSD_GRID_100.num_classes)
    if (class_numbers > max_numbers).any():
        raise ValueError("class_numbers 的元素的值超过了 device_types 允许的上限")

    class_indices = class_numbers.astype(np.intp) - 1
    class_indices[mask_200] += RSD_GRID_100.num_classes
    class_table = _get_class_table()
    class_params = class_table[class_indices, :]  # 返回 copy

    return ClassParams(*(class_params[..., i] for i in range(class_params.shape[-1])))


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
    velocity_center: NDArray[np.float64]
    velocity_width: NDArray[np.float64]
    diameter_center: NDArray[np.float64]
    diameter_width: NDArray[np.float64]


def _pluck(iterable: Iterable[object], name: str) -> list[Any]:
    return [getattr(obj, name) for obj in iterable]


def _safe_concat(
    arrays: Sequence[ArrayLike], dtype: DTypeLike = np.float64
) -> NDArray[Any]:
    if len(arrays) == 0:
        return np.empty(0, dtype=dtype)
    else:
        return np.concatenate(arrays, dtype=dtype)


# TODO: 添加 add_class_params 开关
def rsds_to_dict(rsds: Sequence[RSD]) -> RSDDict:
    """将多个 `RSD` 对象转换成列式字典

    `RSD.to_dict` 方法的批量版本，比循环调用更快。
    """
    # 貌似 pluck 比 attrgetter + zip 要快一点
    repeats = np.array(_pluck(rsds, "num_records"), dtype=np.int64)
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
        "time": _safe_concat(_pluck(rsds, "times"), dtype="datetime64[us]"),
        "sensor_status": np.repeat(
            np.array(_pluck(rsds, "sensor_status"), dtype=np.int64), repeats
        ),
        "device_type": np.repeat(
            np.array(_pluck(rsds, "device_type"), dtype=np.int64), repeats
        ),
        "rain_flag": _safe_concat(_pluck(rsds, "rain_flags"), dtype=np.bool_),
        "class_number": _safe_concat(_pluck(rsds, "class_numbers"), dtype=np.int64),
        "particle_number": _safe_concat(
            _pluck(rsds, "particle_numbers"), dtype=np.int64
        ),
    }

    class_params = lookup_class_params(data["device_type"], data["class_number"])
    data["velocity_center"] = class_params.velocity_centers
    data["velocity_width"] = class_params.velocity_widths
    data["diameter_center"] = class_params.diameter_centers
    data["diameter_width"] = class_params.diameter_widths

    return cast(RSDDict, data)


def rsds_to_dataframe(rsds: Sequence[RSD]) -> pd.DataFrame:
    """将多个 `RSD` 对象转换成 pandas 的 `DataFrame` 对象

    `RSD.to_dataframe` 方法的批量版本，比循环调用更快。
    """
    import pandas as pd

    return pd.DataFrame(rsds_to_dict(rsds))


def resample_rsd_dataframe(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """对 `RSD` 转换得到的 pandas `DataFrame` 进行时间重采样

    要求 `df` 含有 `station_id`、`time`、`class_number`、`rain_flag` 和 `particle_number` 列。
    换句话说最好应用于 `RSD.to_dataframe` 或 `rsds_to_dataframe` 的返回值。

    会按 `station_id` 和 `time` 进行分组，计算 `freq` 时间窗口内 `particle_number` 的和。
    并且能正确处理 `rain_flag=False` 的占位行。
    """
    import pandas as pd

    # 普通列加上时间 grouper 会自动丢弃空时间窗口
    # 提前过滤时间窗口里占位的无雨行
    grouper = pd.Grouper(key="time", freq=freq, closed="right", label="right")  # pyright: ignore[reportCallIssue]
    window_flags = df.groupby(["station_id", grouper])["rain_flag"].transform("any")
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


def build_rsd_dataarray(
    device_type: int,
    class_numbers: ArrayLike,
    particle_numbers: ArrayLike,
    times: ArrayLike | None = None,
) -> xr.DataArray:
    """用雨滴谱数据构造多维的 xarray 的 `DataArray` 对象

    `times=None` 时 `DataArray` 的维度是 `(velocity_center, diameter_center)`；
    否则会按时间分组，维度是 `(time, velocity_center, diameter_center)`。

    元素值是粒子数，`velocity_width` 和 `diameter_width` 是辅助坐标。
    相比 `RSD.to_dataarray` 方法不含元数据。
    """
    import xarray as xr

    if device_type not in {0, 1}:
        raise ValueError(f"{device_type=} 应该是 0 或 1")

    class_numbers = np.atleast_1d(np.asarray(class_numbers))
    if not np.issubdtype(class_numbers.dtype, np.integer):
        raise TypeError("class_numbers 必须是整数类型")
    if (class_numbers < 1).any():
        raise ValueError("class_numbers 的元素的值必须 >= 1")
    rsd_grid = get_rsd_grid(device_type)
    if (class_numbers > rsd_grid.num_classes).any():
        raise ValueError("class_numbers 的元素的值超过了 device_type 允许的上限")

    # 允许 particle_numbers 是浮点数类型
    particle_numbers = np.atleast_1d(np.asarray(particle_numbers))
    if particle_numbers.shape != class_numbers.shape:
        raise ValueError("class_numbers 和 particle_numbers 的形状必须相同")

    coords = {
        "velocity_center": rsd_grid.velocity.centers,
        "diameter_center": rsd_grid.diameter.centers,
        "velocity_width": ("velocity_center", rsd_grid.velocity.widths),
        "diameter_width": ("diameter_center", rsd_grid.diameter.widths),
    }

    class_indices = class_numbers.ravel() - 1

    if times is None:
        indexer = (class_indices,)
        flat_shape = (rsd_grid.num_classes,)
        grid_shape = rsd_grid.shape
        dims = ["velocity_center", "diameter_center"]
    else:
        # TODO: 时间精度截断？
        times = np.atleast_1d(np.asarray(times, dtype="datetime64[us]"))
        if times.shape != class_numbers.shape:
            raise ValueError("times 和 class_numbers 的形状必须相同")

        # np.unique 默认返回排序后的唯一值
        unique_times, time_indices = np.unique(times.ravel(), return_inverse=True)
        indexer = (time_indices, class_indices)
        flat_shape = (len(unique_times), rsd_grid.num_classes)
        grid_shape = (len(unique_times), *rsd_grid.shape)
        dims = ["time", "velocity_center", "diameter_center"]
        coords["time"] = unique_times

    data = np.zeros(flat_shape, dtype=particle_numbers.dtype)
    data[indexer] = particle_numbers.ravel()
    data = data.reshape(grid_shape)
    da = xr.DataArray(data, dims=dims, coords=coords)

    return da


def get_bin_edges(centers: ArrayLike, widths: ArrayLike) -> NDArray[np.float64]:
    """根据分箱中心和宽度计算分箱边缘"""
    centers = np.atleast_1d(np.asarray(centers, dtype=np.float64))
    widths = np.atleast_1d(np.asarray(widths, dtype=np.float64))
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers 必须是 1 维非空数组")
    if len(centers) != len(widths):
        raise ValueError("centers 和 widths 的形状必须相同")

    half_widths = widths / 2
    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[:-1] = centers - half_widths
    edges[-1] = centers[-1] + half_widths[-1]

    return edges


def mass_weighted_diameter(diameters: ArrayLike, particle_numbers: ArrayLike) -> float:
    """计算质量加权平均直径。输入是空数组时会返回 0.0。"""
    # 转换成 float64 避免幂运算溢出
    diameters = np.asarray(diameters, dtype=np.float64)
    particle_numbers = np.asarray(particle_numbers, dtype=np.float64)
    if diameters.shape != particle_numbers.shape:
        raise ValueError("diameters 和 particle_numbers 的形状必须相同")

    # 因为是非负数加和，可以直接跟 0 做比较
    denom = np.sum(particle_numbers * diameters**3)
    if denom == 0:
        return 0.0

    num = np.sum(particle_numbers * diameters**4)
    dm = float(num / denom)

    return dm


def gunn_kinzer_velocity(diameter: ArrayLike) -> NDArray[np.float64]:
    """计算液态雨滴的末速度。输入输出的单位是 mm 和 m/s。"""
    diameter = np.asarray(diameter, dtype=np.float64)
    coeffs = [-0.002362, 0.07934, -0.9551, 4.932, -0.1021]
    velocity = np.polyval(coeffs, diameter)

    return cast(NDArray[np.float64], velocity)
