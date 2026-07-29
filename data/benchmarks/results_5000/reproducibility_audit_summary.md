# Reproducibility Audit Summary

- Generated UTC: 2026-07-29T01:28:38.635832+00:00
- Branch: `thesis/reproducibility-corrections`
- Commit: `d9b7a7ceaca25d80af74fef4d71101681aa32606`
- Python: `3.10.0`
- OS: `Windows-10-10.0.26200-SP0`
- CPU: `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`
- GPU: `not detected`
- RAM bytes: `16936132608`
- Tests passed: **True**
- Artifact validation passed: **True**

## Executed commands

- `pytest services/api/tests -q` -> exit 0
- `python services/api/Scripts/validate_thesis_artifacts.py` -> exit 0

## Test output

```text
........................................................................ [ 44%]
........................................................................ [ 88%]
...................                                                      [100%]
============================== warnings summary ===============================
..\Users\USER\AppData\Local\Programs\Python\Python310\lib\site-packages\starlette\formparsers.py:10
  C:\Users\USER\AppData\Local\Programs\Python\Python310\lib\site-packages\starlette\formparsers.py:10: PendingDeprecationWarning: Please use `import python_multipart` instead.
    import multipart

services/api/tests/test_statistics.py::TestRobustness::test_single_sample
  C:\Users\USER\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\_methods.py:206: RuntimeWarning: Degrees of freedom <= 0 for slice
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,

services/api/tests/test_statistics.py::TestRobustness::test_single_sample
  C:\Users\USER\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\_methods.py:198: RuntimeWarning: invalid value encountered in scalar divide
    ret = ret.dtype.type(ret / rcount)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
163 passed, 3 warnings in 74.73s (0:01:14)
```

The audit records artifact integrity and executed software checks. It does not convert proxy benchmark results into live-application results.
