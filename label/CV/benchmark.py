"""Measure persistent one- versus two-process inference before choosing workers."""
import multiprocessing as mp
import time
import traceback
from PIL import Image
from common import CV, read_json, write_json


def worker(ids, ready, start, finished):
    try:
        import torch
        from backends import SAM2Backend
        torch.set_num_threads(2)
        backend = SAM2Backend(read_json(CV/"config.json"))
        rows = {r["image_id"]:r for r in read_json(CV/"selection.json")["records"]}
        ready.put(True)
        start.wait()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        count = 0
        for figure_id in ids:
            prep = read_json(CV/"preparation/figures"/f"{figure_id}.json")
            image = Image.open(CV/"results/images"/figure_id).convert("RGB")
            panels = [p for p in prep["panels"] if p["applicable"]]
            # One representative panel per figure makes the benchmark bounded.
            for panel in sorted(panels,key=lambda p:(p['bbox'][2]-p['bbox'][0])*(p['bbox'][3]-p['bbox'][1]),reverse=True)[:1]:
                for candidate in backend.segment(image,panel):
                    count += 1
        torch.cuda.synchronize()
        free,total=torch.cuda.mem_get_info()
        finished.put({"ok":True,"seconds":time.perf_counter()-started,"masks":count,"peak_allocated":torch.cuda.max_memory_allocated(),"device_used_at_finish":total-free,"total_device_bytes":total})
    except Exception:
        ready.put(False)
        finished.put({"ok":False,"error":traceback.format_exc()})


def measure(n, ids):
    ctx=mp.get_context("spawn")
    ready,start,finished=ctx.Queue(),ctx.Event(),ctx.Queue()
    ps=[ctx.Process(target=worker,args=(ids[i::n],ready,start,finished)) for i in range(n)]
    for p in ps:p.start()
    load_ok=[ready.get(timeout=180) for _ in ps]
    started=time.perf_counter();start.set()
    outputs=[finished.get(timeout=600) for _ in ps]
    elapsed=time.perf_counter()-started
    for p in ps:p.join(timeout=30)
    result={"workers":n,"elapsed_seconds":elapsed,"ok":all(load_ok) and all(o["ok"] for o in outputs),"processes":outputs}
    print(result,flush=True)
    return result


if __name__=="__main__":
    ids=["2225.png","4647.png","4984.png"]
    one=measure(1,ids);two=measure(2,ids)
    memory_ok=two["ok"] and all(p["device_used_at_finish"]<.85*p["total_device_bytes"] for p in two["processes"])
    workers=2 if memory_ok and two["elapsed_seconds"]<.95*one["elapsed_seconds"] else 1
    report={"ids":ids,"one_worker":one,"two_workers":two,"chosen_workers":workers,"rule":"Choose two only if at least 5% faster, no errors, and less than 85% device memory at completion; per-process peak allocation also recorded."}
    write_json(CV/"results/benchmark.json",report)
    print(f"Chosen GPU workers: {workers}",flush=True)
