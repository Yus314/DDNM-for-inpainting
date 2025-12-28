def test_generator():
    """Test how p_sample_loop_progressive generator works"""
    # Simulate the structure
    def sliding_window_progressive():
        # Do some processing...
        result = {"sample": "final_sliding_window_result"}
        # At the end of p_sample_loop_progressive (non-nine-patch):
        return result  # Line 894
    
    def nine_patch_progressive():
        final_result = "nine_patch_result"
        yield {"sample": final_result}  # Line 704
        return  # Line 705
    
    print("=== Test 1: Sliding window (return) ===")
    sample1 = None
    try:
        for sample_dict in sliding_window_progressive():
            sample1 = sample_dict["sample"]
            print(f"  Got sample: {sample1}")
    except StopIteration as e:
        print(f"  StopIteration raised")
        if hasattr(e, 'value'):
            print(f"  Value: {e.value}")
    print(f"  Final sample1: {sample1}")
    
    print("\n=== Test 2: Nine-patch (yield+return) ===")
    sample2 = None
    for sample_dict in nine_patch_progressive():
        sample2 = sample_dict["sample"]
        print(f"  Got sample: {sample2}")
    print(f"  Final sample2: {sample2}")

test_generator_behavior()
