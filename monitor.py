import os
import time
from os.path import join
import argparse
import yaml
from termcolor import colored
from tabulate import tabulate


def monitor_tasks(sweep_dir: str, interval: int):
    exps = [exp for exp in os.listdir(sweep_dir) if os.path.isdir(join(sweep_dir, exp))]
    last_ipf_step = [None] * len(exps)
    last_update = [None] * len(exps)
    
    while True:
        os.system("clear")
        print(f"Task Monitor for sweep directory {colored(sweep_dir, 'red')}")

        table = []
        for i, exp in enumerate(exps):
            status_file = join(sweep_dir, exp, "status.yaml")
            try:
                with open(status_file, "r") as f:
                    status = yaml.safe_load(f)
            except Exception as e:
                # print(f'Caught exception: {e}')
                status = {"host": "?", "gpu_name": "unknown", "gpu_mem": "unknown", "ipf_step": "?", "time_elapsed": 0, "status": "?"}
            
            # Extract fields from the status line
            host = status["host"]
            gpu_name = status["gpu_name"]
            gpu_mem = status["gpu_mem"]
            ipf_step = status["ipf_step"]
            time_elapsed_total = status["time_elapsed"]
            time_elapsed_str = f"{time_elapsed_total // 3600}h {(time_elapsed_total % 3600) // 60}m {time_elapsed_total % 60}s"
            status = status["status"]

            # Update time elapsed since last update
            if ipf_step != last_ipf_step[i]:
                last_update[i] = time.time()
                last_ipf_step[i] = ipf_step
                elapsed_since_update = "0m 0s"
            else:
                dt = int(time.time() - (last_update[i] or time.time()))
                elapsed_since_update = f"{dt // 60}m {dt % 60}s"

            # Add color based on the status
            if status == "training":
                status_colored = colored("Training", "blue")
            elif status == "completed":
                status_colored = colored("Completed", "green")
            else:
                status_colored = colored("Unknown", "red")

            # Add row to the table
            table.append([exp, host, gpu_name, gpu_mem, ipf_step, time_elapsed_str, elapsed_since_update, status_colored])

        # Print table using tabulate
        print(tabulate(table, headers=["Experiment", "Host", "GPU", "GPU Memory", "IPF Step", "Time Elapsed To Last IPF Step", "Time Elapsed Since Last IPF Step", "Status"], tablefmt="fancy_grid"))
        time.sleep(interval)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Monitoring script")
    parser.add_argument("sweep_dir", help="sweep directory to monitor")
    parser.add_argument("-i", "--interval", default=1, help="interval (s) between each refresh")
    args = parser.parse_args()
    monitor_tasks(args.sweep_dir, args.interval)