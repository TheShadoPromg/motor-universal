# scripts/debug_gx_checkpoint.py
import great_expectations as gx
from pprint import pprint

ctx = gx.get_context()
res = ctx.run_checkpoint(checkpoint_name="derived_dynamic")

print("\n=== CHECKPOINT RESULT ===")
print("overall success:", res["success"])
print("run_id:", res["run_id"])
print("validation_results:", len(res["run_results"]))

failed = []
for key, val in res["run_results"].items():
    success = val["validation_result"]["success"]
    name = val["validation_result"]["meta"]["expectation_suite_name"]
    batch = val["validation_result"]["meta"]["active_batch_definition"]["data_asset_name"]
    if not success:
        failed.append((name, batch, key))
    print(f"- suite={name} | data_asset={batch} | success={success}")
    if not success:
        # imprime los 3 primeros fallos
        eds = val["validation_result"]["results"][:3]
        print("  sample failures:")
        for r in eds:
            if not r["success"]:
                print("   •", r["expectation_config"]["expectation_type"], r["result"])
print("\nFAILED ITEMS:")
pprint(failed)
