import fabric
import asyncio
import os
from os.path import join
import argparse
from secret import USER, PSWD, HOSTS, PROJECT_DIR


active_connections = []

async def distribute_tasks(exp_dir: str):
    exps = [exp for exp in os.listdir(exp_dir) if os.path.isdir(join(exp_dir, exp))]
    n_config = len(exps)
    print(f'Found {n_config} experiments to run in sweep dir {exp_dir}')
    exp2host = {}
    i = 0
    tasks_to_run = []
    for host in HOSTS:
        try:
            exp = exps[i]
            config_file = join(exp_dir, exp, "config.yaml")
            conn = fabric.Connection(host, user=USER, connect_kwargs={"password": PSWD})
            conn.open() # check if we can connect to host
            active_connections.append(conn)
            exp2host[exp] = host
            tasks_to_run.append(run_on_host(host, config_file))
            print(f'Experiment {exp} successfully attributed to host {host}')
            i += 1 # moving on to next task
        except Exception as e:
            print(f'Caught exception on host {host}, trying next server\nException caught: {e}')
        
        if i >= n_config: # all tasks have been distributed
            print("All tasks have been distributed")
            break
    
    if i<n_config:
        print("Some tasks have not been distributed.")

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