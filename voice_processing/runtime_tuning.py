import os


_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_WAIT_POLICY": "PASSIVE",
    "ORT_INTRA_OP_NUM_THREADS": "1",
    "ORT_INTER_OP_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "TORCH_NUM_INTEROP_THREADS": "1",
}


def apply_environment_thread_limits():
    for name, value in _THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _env_int(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def _tune_session_options(options):
    options.inter_op_num_threads = _env_int("ORT_INTER_OP_NUM_THREADS", 1)
    options.intra_op_num_threads = _env_int("ORT_INTRA_OP_NUM_THREADS", 1)
    for key in (
        "session.intra_op.allow_spinning",
        "session.inter_op.allow_spinning",
    ):
        try:
            options.add_session_config_entry(key, "0")
        except Exception:
            pass
    return options


def configure_onnxruntime():
    try:
        import onnxruntime as ort
    except Exception:
        return False

    if getattr(ort, "_orchestrator_runtime_tuned", False):
        return True

    original_session_options = ort.SessionOptions
    original_inference_session = ort.InferenceSession

    def session_options(*args, **kwargs):
        return _tune_session_options(original_session_options(*args, **kwargs))

    def inference_session(*args, **kwargs):
        args = list(args)
        if len(args) >= 2:
            if args[1] is None:
                args[1] = session_options()
        else:
            options = kwargs.get("sess_options")
            if options is None:
                kwargs["sess_options"] = session_options()
        return original_inference_session(*args, **kwargs)

    ort.SessionOptions = session_options
    ort.InferenceSession = inference_session
    ort._orchestrator_runtime_tuned = True
    return True


def configure_native_runtime():
    apply_environment_thread_limits()
    onnx_tuned = configure_onnxruntime()
    print(
        "[runtime] thread limits "
        f"omp={os.environ.get('OMP_NUM_THREADS')} "
        f"openblas={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"ort_intra={os.environ.get('ORT_INTRA_OP_NUM_THREADS')} "
        f"ort_inter={os.environ.get('ORT_INTER_OP_NUM_THREADS')} "
        f"onnx_no_spin={onnx_tuned}",
        flush=True,
    )
