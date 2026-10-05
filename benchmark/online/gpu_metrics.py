"""Sample NVML telemetry. Busy percentages are not bandwidth utilization."""

import statistics
import threading
import time


class GpuMetrics:
    def __init__(self, interval=0.1):
        import pynvml as nv
        self.nv = nv
        nv.nvmlInit()
        self.handle = nv.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval
        self.rows = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        nv, h = self.nv, self.handle
        util = nv.nvmlDeviceGetUtilizationRates(h)
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(h, version=nv.nvmlMemory_v2)
        except (AttributeError, nv.NVMLError_FunctionNotFound, nv.NVMLError_NotSupported):
            mem = nv.nvmlDeviceGetMemoryInfo(h)
        return {"time": time.monotonic(), "gpu_busy_percent": util.gpu,
                "memory_controller_busy_percent": util.memory,
                "vram_used_bytes": mem.used, "vram_total_bytes": mem.total,
                "vram_driver_reserved_bytes": getattr(mem, "reserved", None),
                "power_watts": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                "memory_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                "temperature_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)}

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.rows.append(self._read())
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                break

    def start(self):
        self.rows.append(self._read())
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        self.nv.nvmlShutdown()

    def finish(self, begin, first, end):
        self.stop_event.set()
        self.thread.join()
        try:
            self.rows.append(self._read())
        finally:
            self.nv.nvmlShutdown()
        def summarize(start, stop):
            rows = [r for r in self.rows if start <= r["time"] <= stop]
            if not rows:
                return {"samples": 0}
            result = {"samples": len(rows)}
            for key in ("gpu_busy_percent", "memory_controller_busy_percent", "power_watts",
                        "sm_clock_mhz", "memory_clock_mhz", "temperature_c"):
                result[key + "_mean"] = statistics.mean(r[key] for r in rows)
            peak = max(r["vram_used_bytes"] for r in rows)
            result.update(vram_peak_bytes=peak,
                          vram_peak_percent=100 * peak / rows[0]["vram_total_bytes"])
            return result
        energy = 0.0
        for left, right in zip(self.rows, self.rows[1:]):
            dt = max(0.0, min(end, right["time"]) - max(begin, left["time"]))
            energy += dt * (left["power_watts"] + right["power_watts"]) / 2
        return {"request": summarize(begin, end), "prefill": summarize(begin, first),
                "decode": summarize(first, end), "energy_joules_estimate": energy,
                "sampling_interval_seconds": self.interval, "error": self.error,
                "raw_samples": [{**r, "time": r["time"] - begin} for r in self.rows],
                "notes": "Device-wide NVML rolling-window busy time; not SM occupancy, "
                         "FLOP utilization, DRAM GB/s or measured MBU. Power integration "
                         "includes device baseline; short phases have few samples."}
