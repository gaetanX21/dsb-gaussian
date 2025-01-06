import fabric
import asyncio
import os
from os.path import join
import argparse
from secret import USER, PSWD, AVAILABLE_HOSTS, PROJECT_DIR


def free_hosts(hosts: list[str]):
    """
    Makes the hosts available by removing them from used_hosts.txt
    """
    with open("used_hosts.txt", "r") as f:
        used_hosts = [line.strip("\n") for line in f.readlines()]
    n_used = len(used_hosts)
    used_hosts = [host for host in used_hosts if host not in hosts]
    n_removed = n_used - len(used_hosts)
    with open("used_hosts.txt", "w") as f:
        for host in used_hosts:
            f.write(host+"\n")
    print(f"Removed {n_removed} hosts from used_hosts.txt")

def lock_hosts(hosts: list[str]):
    """
    Maks the hosts unavailable by adding them to used_hosts.txt
    """
    with open("used_hosts.txt", "r") as f:
        used_hosts = [line.strip("\n") for line in f.readlines()]
    n_used = len(used_hosts)
    used_hosts = list(set(used_hosts+hosts)) # quick union
    n_added = len(used_hosts) - n_used
    with open("used_hosts.txt", "w") as f:
        for host in used_hosts:
            f.write(host+"\n")
        print(f'Added {n_added} hosts to used_hosts.txt')


active_connections = []

async def distribute_tasks(exp_dir: str):
    exps = [exp for exp in os.listdir(exp_dir) if os.path.isdir(join(exp_dir, exp))]
    n_config = len(exps)
    print(f'Found {n_config} experiments to run in sweep dir {exp_dir}')
    exp2host = {}
    i = 0
    tasks_to_run = []
    used_hosts = [] # hosts we're going to use
    for host in AVAILABLE_HOSTS:
        try:
            exp = exps[i]
            config_file = join(exp_dir, exp, "config.yaml")
            conn = fabric.Connection(host, user=USER, connect_kwargs={"password": PSWD})
            conn.open() # check if we can connect to host
            active_connections.append(conn)
            exp2host[exp] = host
            tasks_to_run.append(run_on_host(host, config_file))
            print(f'Experiment {exp} successfully attributed to host {host}')
            used_hosts.append(host)
            i += 1 # moving on to next task
        except Exception as e:
            print(f'Caught exception on host {host}, trying next server\nException caught: {e}')
        
        if i >= n_config: # all tasks have been distributed
            print("All tasks have been distributed")
            break
    
    if i<n_config:
        print("Some tasks have not been distributed.")

    lock_hosts(used_hosts)
    await asyncio.gather(*tasks_to_run)


async def run_on_host(host, config_file: str):
    conn = None
    try:
        conn = fabric.Connection(host, user=USER, connect_kwargs={"password": PSWD})
        cmd = f"cd {PROJECT_DIR} && python main.py --config_file {config_file} --host {host}"
        
        # Start the process
        result = await asyncio.to_thread(conn.run, cmd)
        print(f"{host} finished task")
        
    except asyncio.CancelledError:
        print(f"Cancelling task on {host}")
        if conn:
            try:
                # Force kill any python processes started by this user
                conn.run("pkill -f 'python main.py'", warn=True)
            except:
                pass
        raise
    except Exception as e:
        print(f"Caught exception on {host}: {e}")
    finally:
        if conn:
            free_hosts([host])
            conn.close()



async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="directory with experiments")
    args = parser.parse_args()
    
    try:
        await distribute_tasks(args.exp_dir)
    except asyncio.CancelledError:
        print("Tasks cancelled")
    finally:
        # Ensure all connections are closed
        for conn in active_connections:
            try:
                conn.close()
            except:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Forced shutdown')