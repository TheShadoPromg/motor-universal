import pandas as pd, great_expectations as gx

PATH = "data/raw/eventos_numericos.csv"
SUITE = "eventos_numericos_suite"

df = pd.read_csv(PATH, dtype={"numero": str, "e_pos1": int, "e_pos2": int, "e_pos3": int})
ctx = gx.get_context()
try: suite = ctx.get_expectation_suite(SUITE)
except Exception: suite = ctx.add_or_update_expectation_suite(SUITE)

v = ctx.get_validator(
    datasource_name="filesystem",
    data_connector_name="runtime_data_connector",
    data_asset_name="eventos_numericos",
    runtime_parameters={"batch_data": df},
    batch_identifiers={"default_identifier_name": "eventos"},
    expectation_suite=suite,
)

# columnas exactas
v.expect_table_columns_to_match_set(["fecha","numero","e_pos1","e_pos2","e_pos3"])
# filas > 0
v.expect_table_row_count_to_be_between(min_value=1)
# tipos y dominios
v.expect_column_values_to_not_be_null("fecha", mostly=1.0)
v.expect_column_values_to_match_regex("numero", regex=r"^\d{2}$", mostly=1.0)
for c in ["e_pos1","e_pos2","e_pos3"]:
    v.expect_column_values_to_be_in_set(c, value_set=[0,1], mostly=1.0)
# (opcional) por fecha deben existir 100 números
# esto se verifica en el transform; aquí dejamos reglas generales

suite_obj = v.get_expectation_suite()
suite_obj.expectation_suite_name = SUITE
ctx.add_or_update_expectation_suite(expectation_suite=suite_obj)
ctx.build_data_docs()
print(f"✅ Suite '{SUITE}' creada/actualizada.")