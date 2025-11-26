# ejecuta esto en un script o notebook dentro de tu venv
import great_expectations as gx
context = gx.get_context(mode="file")  # crea/usa un File Data Context en el cwd
print(type(context).__name__)
"""Bootstrap mínimo para Great Expectations; permite inicializar contexto GE desde CLI."""
