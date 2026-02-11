"""多线程读取一个站点的所有 BUFR 文件，重采样成逐小时累积的表格，导出成 parquet 文件。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

import cnrsd


def make_rsd_pattern(station_id: str) -> str:
    return f"Z_SURF_I_{station_id}_*_O_AWS-RSD-MM_FTM.BIN"


def main() -> None:
    # dirpath = Path("data")
    dirpath = Path("data")
    station_id = "53691"
    pattern = make_rsd_pattern(station_id)

    # 多线程读取 bytes，单线程解析 bytes 成 RSD 对象
    rsds: list[cnrsd.RSD] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(Path.read_bytes, filepath): filepath
            for filepath in dirpath.rglob(pattern)
        }
        for future in as_completed(futures):
            try:
                data = future.result()
                rsds.append(cnrsd.RSD.from_bytes(data))
            except cnrsd.RSDError:
                logger.exception(futures[future])

    # 从逐分钟重采样到逐小时
    df = cnrsd.rsds_to_dataframe(rsds)
    df = cnrsd.resample_rsd_dataframe(df, freq="h")

    # 导出 parquet 格式
    # 这里为了演示，不保存文件
    df.to_parquet(index=False, compression="zstd")
    logger.info("read_files done")


if __name__ == "__main__":
    main()
