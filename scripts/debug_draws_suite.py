import great_expectations as gx

ctx = gx.get_context()
res = ctx.run_checkpoint(checkpoint_name="derived_dynamic")

for key, val in res["run_results"].items():
    suite_name = val["validation_result"]["meta"]["expectation_suite_name"]
    if suite_name == "draws_suite":
        print(f"\n🔍 Suite: {suite_name}")
        print("Success:", val["validation_result"]["success"])
        for r in val["validation_result"]["results"]:
            if not r["success"]:
                e_type = r["expectation_config"]["expectation_type"]
                cols = r["expectation_config"]["kwargs"]
                print(f"❌ Expectation failed: {e_type}")
                print("   → Details:", cols)
