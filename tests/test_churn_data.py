import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from churn_data import load_data


class LoadDataTest(unittest.TestCase):
    def test_features_use_only_events_available_before_cutoff(self):
        header = "record_id,subscriber_key,event_ts,ingest_ts,duration_sec,bytes_down\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("out/reference/passport.json", json.dumps({
                    "period_start": "2025-10-01", "period_days": 4, "seed": 42,
                }))
                archive.writestr(
                    "out/reference/churn_truth.csv",
                    "subscriber_key,churn_day_index\nA,2\nB,-1\nC,1\nD,-1\n",
                )
                archive.writestr("out/raw/cdr/dt=2025-10-01/cdr.csv", header + (
                    "1,A,2025-10-01 10:00:00,2025-10-01 11:00:00,10,100\n"
                    "2,B,2025-10-01 10:00:00,2025-10-01 11:00:00,10,100\n"
                    "3,C,2025-10-01 10:00:00,2025-10-01 11:00:00,10,100\n"
                ))
                archive.writestr("out/raw/cdr/dt=2025-10-02/cdr.csv", header + (
                    "4,A,02.10.2025 10:00:00,2025-10-02 11:00:00,-5,200\n"
                    "4,A,02.10.2025 10:00:00,2025-10-02 11:00:00,-5,200\n"
                    "5,A,2025-10-02 12:00:00,2025-10-02 13:00:00,10,300\n"
                    "6,A,2025-10-02 14:00:00,2025-10-03 00:00:00,30,999\n"
                    "7,A,2025-10-03 01:00:00,2025-10-03 02:00:00,30,999\n"
                ))
                archive.writestr("out/raw/cdr/dt=2025-10-03/cdr.csv", header + (
                    "8,D,2025-10-03 10:00:00,2025-10-03 11:00:00,10,999\n"
                ))
            table, quality = load_data(path, {
                "cutoff": "2025-10-03",
                "observation_days": 2,
                "recent_days": 1,
                "horizon_days": 2,
            })

        self.assertEqual(table["subscriber_key"].tolist(), ["A", "B"])
        self.assertEqual(table["churned"].tolist(), [1, 0])
        first = table.iloc[0]
        self.assertEqual(first["event_count"], 3)
        self.assertEqual(first["active_days"], 2)
        self.assertEqual(first["days_since_last_event"], 1)
        self.assertEqual(first["recent_event_count"], 2)
        self.assertEqual(first["total_bytes"], 600)
        self.assertEqual(first["total_duration"], 20)
        self.assertEqual(quality["duplicate_rows"], 1)
        self.assertEqual(quality["invalid_rows"], 1)
        self.assertEqual(quality["received_at_or_after_cutoff"], 1)
        self.assertEqual(quality["excluded_prior_churn_with_cdr"], 1)
        self.assertEqual(quality["eligible_without_cdr"], 1)
