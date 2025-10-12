# ejecuta esto en un script o notebook dentro de tu venv
import great_expectations as gx
context = gx.get_context(mode="file")  # crea/usa un File Data Context en el cwd
print(type(context).__name__)