# scripts/build_events_suite.py
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import great_expectations as gx
from engine._utils.schema import normalize_events_df

PATH = REPO_ROOT / "data" / "raw" / "eventos_numericos.csv"
SUITE = "events_suite"

df_raw = pd.read_csv(PATH)
df = normalize_events_df(df_raw)  # <-- handles long or wide

ctx = gx.get_context()
try:
    suite = ctx.get_expectation_suite(SUITE)
except Exception:
    suite = ctx.add_or_update_expectation_suite(SUITE)

v = ctx.get_validator(
    datasource_name="filesystem",
    data_connector_name="runtime_data_connector",
    data_asset_name="events",
    runtime_parameters={"batch_data": df},
    batch_identifiers={"default_identifier_name": "events"},
    expectation_suite=suite,
)

v.expect_table_columns_to_match_set(["date","number","pos1","pos2","pos3"])
v.expect_table_row_count_to_be_between(min_value=1)
v.expect_column_values_to_not_be_null("date")
v.expect_column_values_to_match_regex("number", regex=r"^\d{2}$", mostly=1.0)
for c in ["pos1","pos2","pos3"]:
    v.expect_column_values_to_be_in_set(c, value_set=[0,1], mostly=1.0)

suite_obj = v.get_expectation_suite()
suite_obj.expectation_suite_name = SUITE
ctx.add_or_update_expectation_suite(expectation_suite=suite_obj)
ctx.build_data_docs()
print("✅ Built/updated suite 'events_suite' (long/wide supported).")
