import socket
import time
import threading
import argparse
import sys
import urllib.request
import urllib.error

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Optionally open and close a port on the local node.")
parser.add_argument("--local-ip", required=False, help="Local IP address to bind the server.")
parser.add_argument("--local-port", type=int, required=False, help="Port number to bind the server.")
parser.add_argument("--enable-port", action="store_true", help="Enable opening and closing of local port.")
parser.add_argument("--node-ips", required=True, help="Comma-separated list of node IPs.")
parser.add_argument("--node-ports", required=True, help="Comma-separated list of ports to check.")
parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for waiting on all ports (default: 600s / 10 minutes). Set to 0 for no timeout.")
parser.add_argument("--wait-for-all-ports", action="store_true", help="Wait until all node ports are open (TCP).")
parser.add_argument("--wait-for-all-health", action="store_true", help="Wait until http://ip:port/health returns 200 for all nodes.")
parser.add_argument("--health-endpoint", default="/health", help="Path for health check (default: /health).")
args = parser.parse_args()

# Parse node IPs and ports from command-line arguments
NODE_IPS = [ip.strip() for ip in args.node_ips.split(",") if ip.strip()]
NODE_PORTS = [int(port.strip()) for port in args.node_ports.split(",") if port.strip()]

# Ensure port list matches node list or default to using the same port for all nodes
if len(NODE_PORTS) == 1:
    NODE_PORTS *= len(NODE_IPS)
elif len(NODE_PORTS) != len(NODE_IPS):
    print("Error: Number of ports must match number of node IPs or only one port should be given for all.")
    exit(1)

server_socket = None  # Global server socket reference

def is_port_open(ip, port):
    """Check if a given IP and port are accessible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)  # Avoid long wait times
        return s.connect_ex((ip, port)) == 0

def wait_for_all_ports():
    """Wait until all nodes have opened the specified ports, with optional timeout."""
    start_time = time.time()
    timeout = args.timeout

    while True:
        # Check for timeout (if timeout > 0)
        if timeout > 0:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                # Report which nodes/ports are still not responding
                not_open = [(ip, port) for ip, port in zip(NODE_IPS, NODE_PORTS) if not is_port_open(ip, port)]
                print(f"ERROR: Timeout after {timeout} seconds waiting for ports to open.", flush=True)
                print(f"The following nodes/ports are still not responding:", flush=True)
                for ip, port in not_open:
                    print(f"  - {ip}:{port}", flush=True)
                sys.exit(1)

        all_open = all(is_port_open(ip, port) for ip, port in zip(NODE_IPS, NODE_PORTS))
        if all_open:
            break

        if timeout > 0:
            remaining = timeout - (time.time() - start_time)
            print(f"Waiting for nodes.{NODE_PORTS},{NODE_IPS} . . ({remaining:.0f}s remaining)", flush=True)
        else:
            print(f"Waiting for nodes.{NODE_PORTS},{NODE_IPS} . .", flush=True)
        time.sleep(5)


def check_health(ip, port, path="/health", timeout=2):
    """Return True if http://ip:port/path returns HTTP 200."""
    try:
        url = f"http://{ip}:{port}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200) == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def wait_for_all_health(path="/health"):
    """Wait until http://hostip:port/path returns 200 for all (NODE_IPS, NODE_PORTS) within timeout."""
    start_time = time.time()
    timeout_sec = args.timeout
    check_timeout = 2

    while True:
        if timeout_sec > 0:
            elapsed = time.time() - start_time
            if elapsed >= timeout_sec:
                not_ready = [
                    (ip, port)
                    for ip, port in zip(NODE_IPS, NODE_PORTS)
                    if not check_health(ip, port, path, check_timeout)
                ]
                print(f"ERROR: Timeout after {timeout_sec} seconds waiting for health endpoints.", flush=True)
                print(f"The following (http://ip:port{path}) are still not responding:", flush=True)
                for ip, port in not_ready:
                    print(f"  - http://{ip}:{port}{path}", flush=True)
                sys.exit(1)

        all_ready = all(
            check_health(ip, port, path, check_timeout)
            for ip, port in zip(NODE_IPS, NODE_PORTS)
        )
        if all_ready:
            break

        if timeout_sec > 0:
            remaining = timeout_sec - (time.time() - start_time)
            print(
                f"Waiting for health on {list(zip(NODE_IPS, NODE_PORTS))} ({path}) .. ({remaining:.0f}s remaining)",
                flush=True,
            )
        else:
            print(f"Waiting for health on {list(zip(NODE_IPS, NODE_PORTS))} ({path}) ..", flush=True)
        time.sleep(5)


def open_port():
    """Open a listening socket on the current node."""
    global server_socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.local_ip, args.local_port))
    server_socket.listen(5)
    print(f"Port {args.local_port} is now open on {args.local_ip}.")
    while True:
        conn, addr = server_socket.accept()
        conn.close()

def close_port():
    """Close the opened port."""
    global server_socket
    if server_socket:
        server_socket.close()
        print(f"Port {args.local_port} has been closed on {args.local_ip}.")

if __name__ == "__main__":
    if not NODE_IPS:
        print("Error: NODE_IPS argument is empty or not set.")
        exit(1)

    if args.enable_port:
        threading.Thread(target=open_port, daemon=True).start()
    
    if args.wait_for_all_ports:
        wait_for_all_ports()

    if args.wait_for_all_health and args.health_endpoint:
        wait_for_all_health(args.health_endpoint)

    if args.enable_port:
        time.sleep(30)
        close_port()