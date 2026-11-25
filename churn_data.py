import json
import logging
from collections import Counter
from zipfile import ZipFile

import pandas as pd


LOGGER = logging.getLogger(__name__)
CDR_COLUMNS = (
    "record_id",
    "subscriber_key",
    "event_ts",
    "ingest_ts",
    "duration_sec",
    "bytes_down",
)
FEATURE_COLUMNS = (
    "event_count",
    "active_days",
    "total_duration",
    "total_bytes",
    "days_since_last_event",
    "recent_event_count",
    "recent_bytes",
    "activity_ratio",
)


def parse_timestamp(values):
    iso = values.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", na=False)
    european = values.str.fullmatch(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}", na=False)
    parsed = pd.to_datetime(
        values.where(iso), format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    parsed.loc[european] = pd.to_datetime(
        values.loc[european], format="%d.%m.%Y %H:%M:%S", errors="coerce"
    )
    return parsed


def aggregate_day(records, day, cutoff, recent_start, counts):
    counts["rows_read"] += len(records)
    event = parse_timestamp(records["event_ts"])
    ingest = parse_timestamp(records["ingest_ts"])
    valid = (
        records["subscriber_key"].notna()
        & records["record_id"].notna()
        & records["duration_sec"].notna()
        & records["bytes_down"].notna()
        & event.ge(day)
        & event.lt(day + pd.Timedelta(days=1))
        & ingest.ge(event)
    )
    counts["invalid_rows"] += int((~valid).sum())
    counts["received_at_or_after_cutoff"] += int((valid & ingest.ge(cutoff)).sum())
    available = records.loc[valid & ingest.lt(cutoff)].drop_duplicates().copy()
    counts["duplicate_rows"] += (
        int((valid & ingest.lt(cutoff)).sum()) - len(available)
    )
    if available["record_id"].duplicated().any():
        raise ValueError(f"Conflicting CDR record IDs in partition {day.date()}")

    counts["accepted_rows"] += len(available)
    counts["negative_duration_rows"] += int(available["duration_sec"].lt(0).sum())
    counts["negative_bytes_rows"] += int(available["bytes_down"].lt(0).sum())
    available["duration_sec"] = available["duration_sec"].clip(lower=0)
    available["bytes_down"] = available["bytes_down"].clip(lower=0)
    daily = available.groupby("subscriber_key", sort=False).agg(
        event_count=("record_id", "size"),
        total_duration=("duration_sec", "sum"),
        total_bytes=("bytes_down", "sum"),
    )
    daily["active_days"] = 1
    daily["last_day"] = day
    daily["recent_event_count"] = daily["event_count"] if day >= recent_start else 0
    daily["recent_bytes"] = daily["total_bytes"] if day >= recent_start else 0
    return daily


def load_data(archive_path, config):
    cutoff = pd.Timestamp(config["cutoff"])
    observation_days = config["observation_days"]
    recent_days = config["recent_days"]
    horizon_days = config["horizon_days"]
    if not 0 < recent_days < observation_days or horizon_days < 1:
        raise ValueError("Set 0 < recent_days < observation_days and horizon_days >= 1")
    if cutoff != cutoff.normalize() or cutoff.tz is not None:
        raise ValueError("cutoff must be a calendar date without a time zone")

    start = cutoff - pd.Timedelta(days=observation_days)
    recent_start = cutoff - pd.Timedelta(days=recent_days)
    end = cutoff + pd.Timedelta(days=horizon_days)
    counts = Counter()
    missing_days = []
    unreadable_days = []
    daily_tables = []
    with ZipFile(archive_path) as archive:
        passport = json.loads(archive.read("out/reference/passport.json"))
        period_start = pd.Timestamp(passport["period_start"])
        period_end = period_start + pd.Timedelta(days=passport["period_days"])
        if not period_start <= start < cutoff < end <= period_end:
            raise ValueError("Observation and forecast windows must fit within the dataset")

        with archive.open("out/reference/churn_truth.csv") as source:
            truth = pd.read_csv(
                source, usecols=["subscriber_key", "churn_day_index"],
                dtype={"subscriber_key": "string", "churn_day_index": "int64"},
            )
        partitions = set(archive.namelist())
        for day in pd.date_range(start, cutoff, inclusive="left"):
            date = day.date().isoformat()
            partition = f"out/raw/cdr/dt={date}/cdr.csv"
            if partition not in partitions:
                missing_days.append(date)
                LOGGER.warning("Missing CDR partition for %s", date)
                continue
            try:
                with archive.open(partition) as source:
                    records = pd.read_csv(
                        source, usecols=CDR_COLUMNS,
                        dtype={"subscriber_key": "string", "event_ts": "string", "ingest_ts": "string"},
                    )
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                unreadable_days.append(date)
                LOGGER.warning("Unreadable CDR partition %s: %s", date, exc)
                continue
            daily_tables.append(aggregate_day(records, day, cutoff, recent_start, counts))
            LOGGER.debug("Read CDR partition %s (%d rows)", date, len(records))

    if not counts["accepted_rows"]:
        raise ValueError("No usable CDR observations before cutoff")
    if truth["subscriber_key"].isna().any() or truth["subscriber_key"].duplicated().any():
        raise ValueError("Missing or duplicate subscriber keys in churn_truth.csv")
    if not truth["churn_day_index"].between(-1, passport["period_days"] - 1).all():
        raise ValueError("churn_day_index outside the dataset period")

    activity = pd.concat(daily_tables).groupby(level="subscriber_key").agg(
        event_count=("event_count", "sum"),
        active_days=("active_days", "sum"),
        total_duration=("total_duration", "sum"),
        total_bytes=("total_bytes", "sum"),
        last_day=("last_day", "max"),
        recent_event_count=("recent_event_count", "sum"),
        recent_bytes=("recent_bytes", "sum"),
    ).reset_index()
    activity["days_since_last_event"] = (cutoff - activity.pop("last_day")).dt.days
    previous_count = activity["event_count"] - activity["recent_event_count"]
    activity["activity_ratio"] = (
        (activity["recent_event_count"] + 1) / recent_days
        / ((previous_count + 1) / (observation_days - recent_days))
    )

    cutoff_index = (cutoff - period_start).days
    end_index = (end - period_start).days
    churn_days = truth["churn_day_index"]
    truth["eligible"] = churn_days.eq(-1) | churn_days.ge(cutoff_index)
    truth["churned"] = churn_days.between(cutoff_index, end_index - 1).astype("int8")
    joined = activity.merge(truth, on="subscriber_key", how="left", validate="one_to_one")
    if joined["eligible"].isna().any():
        raise ValueError("CDR contains subscriber keys absent from churn_truth.csv")
    table = joined.loc[joined["eligible"], ["subscriber_key", *FEATURE_COLUMNS, "churned"]]
    table = table.reset_index(drop=True)
    if not table[FEATURE_COLUMNS[0]].ge(table["active_days"]).all():
        raise ValueError("Active days exceed event count")
    if table["churned"].nunique() != 2:
        raise ValueError("Both churn classes must be present among observed subscribers")

    quality = {
        "period_start": start.date().isoformat(),
        "cutoff": cutoff.date().isoformat(),
        "forecast_end_exclusive": end.date().isoformat(),
        "dataset_seed": passport.get("seed"),
        "missing_cdr_days": missing_days,
        "unreadable_cdr_days": unreadable_days,
        "rows_read": counts["rows_read"],
        "invalid_rows": counts["invalid_rows"],
        "received_at_or_after_cutoff": counts["received_at_or_after_cutoff"],
        "duplicate_rows": counts["duplicate_rows"],
        "accepted_rows": counts["accepted_rows"],
        "negative_duration_rows": counts["negative_duration_rows"],
        "negative_bytes_rows": counts["negative_bytes_rows"],
        "subscribers_with_cdr": len(activity),
        "excluded_prior_churn_with_cdr": int((~joined["eligible"]).sum()),
        "eligible_without_cdr": int(truth["eligible"].sum()) - len(table),
        "training_table_rows": len(table),
        "positive_labels": int(table["churned"].sum()),
    }
    LOGGER.info(
        "Accepted %d of %d CDR rows. Feature table has %d subscribers (%d with churn)",
        quality["accepted_rows"], quality["rows_read"], len(table), quality["positive_labels"],
    )
    return table, quality
