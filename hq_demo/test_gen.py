# Test how generator works with return
def sliding_window_gen():
    result = {"sample": "final_result"}
    return result

def nine_patch_gen():
    result = "nine_patch_result"
    yield {"sample": result}
    return

print("=== Sliding window (return) ===")
s1 = None
for sample_dict in sliding_window_gen():
    s1 = sample_dict["sample"]
    print(f"Got: {s1}")
print(f"Final s1: {s1}")

print("\n=== Nine-patch (yield) ===")
s2 = None
for sample_dict in nine_patch_gen():
    s2 = sample_dict["sample"]
    print(f"Got: {s2}")
print(f"Final s2: {s2}")
