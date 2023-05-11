try:
    import meme
except (ImportError, ModuleNotFoundError):
    pass
from datetime import datetime


dates = [
    (
        datetime(2021, 4, 14, 00, 00, 00),
        datetime(2021, 4, 14, 23, 59, 59),
    ),
    (
        datetime(2021, 4, 15, 00, 00, 00),
        datetime(2021, 4, 15, 23, 59, 59),
    ),
    (
        datetime(2021, 6, 8, 00, 00, 00),
        datetime(2021, 6, 8, 23, 59, 59),
    ),
    (
        datetime(2021, 6, 9, 00, 00, 00),
        datetime(2021, 6, 9, 23, 59, 59),
    ),
    (
        datetime(2021, 9, 25, 00, 00, 00),
        datetime(2021, 9, 25, 23, 59, 59),
    ),
    (
        datetime(2021, 9, 26, 00, 00, 00),
        datetime(2021, 9, 26, 23, 59, 59),
    ),
    (
        datetime(2021, 11, 13, 00, 00, 00),
        datetime(2021, 11, 13, 23, 59, 59),
    ),
    (
        datetime(2021, 11, 14, 00, 00, 00),
        datetime(2021, 11, 14, 23, 59, 59),
    ),
    (
        datetime(2021, 11, 15, 00, 00, 00),
        datetime(2021, 11, 15, 23, 59, 59),
    ),
    (
        datetime(2021, 11, 16, 00, 00, 00),
        datetime(2021, 11, 16, 23, 59, 59),
    ),
    (
        datetime(2021, 11, 17, 00, 00, 00),
        datetime(2021, 11, 17, 23, 59, 59),
    ),
    (
        datetime(2021, 11, 18, 00, 00, 00),
        datetime(2021, 11, 18, 23, 59, 59),
    ),
    (
        datetime(2021, 11, 19, 00, 00, 00),
        datetime(2021, 11, 19, 23, 59, 59),
    ),
    (
        datetime(2021, 12, 9, 00, 00, 00),
        datetime(2021, 12, 9, 23, 59, 59),
    ),
    (
        datetime(2021, 12, 10, 00, 00, 00),
        datetime(2021, 12, 10, 23, 59, 59),
    ),
]
pvs = [
    "SOLN:IN20:121:BACT",
    "QUAD:IN20:121:BACT",
    "QUAD:IN20:122:BACT",
    "ACCL:IN20:300:L0A_PDES",
    "ACCL:IN20:400:L0B_PDES",
    "ACCL:IN20:300:L0A_ADES",
    "ACCL:IN20:400:L0B_ADES",
    "QUAD:IN20:361:BACT",
    "QUAD:IN20:371:BACT",
    "QUAD:IN20:425:BACT",
    "QUAD:IN20:441:BACT",
    "QUAD:IN20:511:BACT",
    "QUAD:IN20:525:BACT",
    "FBCK:BCI0:1:CHRG_S",
    "OTRS:IN20:571:XRMS",  # OTR2, OTR3 = 621
    "OTRS:IN20:571:YRMS",
    "CAMR:IN20:186:YRMS",
    "CAMR:IN20:186:XRMS",
]
for pv in pvs:
    if "BACT" in pv:
        pvs.append(pv.replace("BACT", "BCTRL"))

print(len(pvs))
for start, end in dates:
    print(str(start), str(end))
    try:
        result = meme.archive.get_dataframe(pvs, start, end, timeout=120)
        result = result[str(start) : str(end)]

        control_pvs = [pv for pv in pvs if "BCTRL" in pv]
        control_data = result[control_pvs].drop_duplicates()
        all_data = result.loc[control_data.index]

        if len(all_data) > 0:
            all_data.to_pickle(f"archive_data/injector_{str(start.date())}.pkl")
    except NameError:
        pass
