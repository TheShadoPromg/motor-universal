import pandas as pd
import great_expectations as gx
import re

PATH = "data/raw/draws.csv"
SUITE = "draws_suite"

df = pd.read_csv(PATH)
ctx = gx.get_context()

try:
    suite = ctx.get_expectation_suite(SUITE)
except Exception:
    suite = ctx.add_or_update_expectation_suite(SUITE)

cols = list(df.columns)
validator = ctx.get_validator(
    datasource_name="filesystem",
    data_connector_name="runtime_data_connector",
    data_asset_name="draws",
    runtime_parameters={"batch_data": df},
    batch_identifiers={"default_identifier_name": "draws"},
    expectation_suite=suite,
)

validator.expect_table_row_count_to_be_between(min_value=1)
validator.expect_table_columns_to_match_set(cols)

# Detecta columna de fecha
fecha_col = next((c for c in cols if c.lower() == "fecha"), None)
if fecha_col:
    validator.expect_column_values_to_not_be_null(fecha_col, mostly=0.99)

# Validación flexible para posiciones
pattern = r"^\d{1,2}$"  # acepta 0–99 con 1 o 2 dígitos
for c in [col for col in cols if col.lower() in {"pos1", "pos2", "pos3"}]:
    validator.expect_column_values_to_not_be_null(c, mostly=0.99)
    validator.expect_column_values_to_match_regex(c, regex=pattern, mostly=0.99)

suite_obj = validator.get_expectation_suite()
suite_obj.expectation_suite_name = SUITE  # <--- línea clave
ctx.add_or_update_expectation_suite(expectation_suite=suite_obj)
ctx.build_data_docs()
print(f"✅ Suite '{SUITE}' actualizada (validación 0–99 tolerante).")
