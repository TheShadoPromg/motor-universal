import os
import yaml
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.core.expectation_suite import ExpectationSuite

# Rutas base
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CTX_DIR = os.path.join(ROOT, "great_expectations")
EXP_DIR = os.path.join(CTX_DIR, "expectations")

# Asegura cwd en la raíz del repo
os.chdir(ROOT)

def load_yaml_suite(context, yaml_path: str) -> None:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No existe el archivo: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    suite_name = data["expectation_suite_name"]
    items = data.get("expectations", [])
    meta = data.get("meta", {})

    # Construir una suite en memoria
    suite = ExpectationSuite(expectation_suite_name=suite_name, expectations=[], meta=meta)

    # Agregar cada expectativa usando ExpectationConfiguration
    for item in items:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Fila inválida en {yaml_path}: {item}")
        exp_type, kwargs = next(iter(item.items()))
        exp_cfg = ExpectationConfiguration(
            expectation_type=exp_type,
            kwargs=kwargs or {},
            meta=None,
        )
        suite.add_expectation(expectation_configuration=exp_cfg)

    # Guardar / registrar en el store del Data Context
    context.add_or_update_expectation_suite(expectation_suite=suite)
    print(f"[OK] Registrada suite: {suite_name}")

def main():
    # Carga el Data Context desde great_expectations/great_expectations.yml
    context = gx.get_context()

    # Registrar ambas suites desde YAML (ya las tienes en great_expectations/expectations/)
    load_yaml_suite(context, os.path.join(EXP_DIR, "draws_suite.yaml"))
    load_yaml_suite(context, os.path.join(EXP_DIR, "derived_dynamic_suite.yaml"))

if __name__ == "__main__":
    main()
"""Registra suites GE en el contexto (draws/events/otros) para ejecución posterior."""
