# Z Image Debugging - Test Recommendations

## Finding Summary

### ✅ FIXES APPLIED
1. **rope_theta = 256.0** (was 10000.0) - CRITICAL FIX
2. **time_scale = 1000.0** (was 1.0) - CRITICAL FIX

### 📊 RESULTS FROM ZIMAGE_8

**Positive Changes:**
- norm(x) now DECREASES: 478→360→366→456 (before: was increasing 480→736)
- Model output is reasonable: norm=620-740
- No more divergence in early steps

**Remaining Issue:**
- Scheduler has HUGE jump at final step: sigma 0.334→0.0003
- This causes norm to spike back up in steps 6-8

### 🧪 RECOMMENDED TESTS

**Test 1: More Steps (Recommended)**
- Use 20 steps instead of 8
- This will create smoother sigma transitions
- Should eliminate the large jump problem

**Test 2: Adjust Shift (Alternative)**
- Try shift=1.0 instead of 3.0
- Makes sigma distribution more linear
- May help with the final step jump

**Test 3: Different Scheduler (If above fail)**
- Try "normal" scheduler instead of "simple"
- May provide better sigma spacing

### 📝 HOW TO TEST

**Option A: Via UI**
1. Keep same prompt and settings
2. Change steps from 8 to 20
3. Generate zimage_9

**Option B: Test Script**
Run from root directory:
```powershell
# Test with 20 steps
python -c "
import sys
sys.path.insert(0, '.')
# Your test code here with steps=20
"
```

### 🎯 EXPECTED OUTCOME

With 20 steps, the sigma schedule will be:
```
sigma: 1.0 → 0.95 → 0.90 → ... → 0.05 → 0.0
```

This should:
1. Maintain smooth norm(x) decrease throughout
2. Avoid the large jump at the end
3. Produce cleaner final image

### 📌 NEXT STEPS

1. **Generate zimage_9 with 20 steps**
2. Check if norm(x) decreases smoothly to ~100-200
3. Verify image has structure (not pure noise)

If zimage_9 still shows noise, the issue may be elsewhere (but logs suggest it should work now).
