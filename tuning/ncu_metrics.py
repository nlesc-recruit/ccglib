from kernel_tuner.observers.ncu import NCUObserver


def get_ncu_observer():
    ncu_metrics = [
        "dram__bytes.sum",  # number of bytes accessed in DRAM
        "dram__bytes_read.sum",  # number of bytes read from DRAM
        "dram__bytes_write.sum",  # number of bytes written to DRAM
    ]

    return NCUObserver(metrics=ncu_metrics)


def get_ncu_metrics():
    metrics = dict()
    metrics["dram_bytes_total"] = lambda p: p["dram__bytes.sum"]
    metrics["dram_bytes_read"] = lambda p: p["dram__bytes_read.sum"]
    metrics["dram_bytes_write"] = lambda p: p["dram__bytes_write.sum"]
    return metrics
