# scripts/update_gx.py
import pandas as pd
import great_expectations as gx

DF_PATH = "data/outputs/derived_dynamic.parquet"
SUITE_NAME = "derived_dynamic_suite"

def main():
    df = pd.read_parquet(DF_PATH)
    ctx = gx.get_context()

    # crea o toma la suite
    try:
        suite = ctx.get_expectation_suite(SUITE_NAME)
    except Exception:
        suite = ctx.add_or_update_expectation_suite(SUITE_NAME)

    validator = ctx.get_validator(
        datasource_name="filesystem",
        data_connector_name="runtime_data_connector",
        data_asset_name="derived_dynamic",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "dd"},
        expectation_suite=suite,
    )

    # --- Expectativas compatibles (v1):
    # filas > 0 => usa "between" con min 1
    validator.expect_table_row_count_to_be_between(min_value=1)

    # columnas exactas
    expected_cols = list(df.columns)
    validator.expect_table_columns_to_match_set(expected_cols)

    # columnas clave no nulas
    for col in ["fecha", "numero", "tipo_relacion", "lag", "oportunidades", "activaciones", "consistencia"]:
        if col in df.columns:
            validator.expect_column_values_to_not_be_null(col)

    # tipos/rangos robustos
    if "consistencia" in df.columns:
        validator.expect_column_values_to_be_between("consistencia", min_value=0.0, max_value=1.0, mostly=1.0)
    if "lag" in df.columns:
        validator.expect_column_values_to_be_between("lag", min_value=1, max_value=3650, mostly=1.0)
    if "oportunidades" in df.columns:
        validator.expect_column_values_to_be_between("oportunidades", min_value=0, mostly=1.0)
    if "activaciones" in df.columns:
        validator.expect_column_values_to_be_between("activaciones", min_value=0, mostly=1.0)

    # guarda suite y build docs
    ctx.add_or_update_expectation_suite(expectation_suite=validator.get_expectation_suite())
    ctx.build_data_docs()
    print("✅ Suite 'derived_dynamic_suite' actualizada y data docs generados.")

if __name__ == "__main__":
    main()