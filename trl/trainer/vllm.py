import asyncio
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import IO, Any, Optional

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient


@dataclass
class vLLM:
    client: AsyncOpenAI
    max_concurrent_tokens: int
    model: str
    process: asyncio.subprocess.Process


async def start_vllm(
    model: str,
    env: Optional[dict[str, str]] = None,
    log_file: str = "./logs/vllm.log",
    max_concurrent_requests: int = 128,
    named_arguments: dict[str, Any] = {},
    timeout: float = 120.0,
    verbosity: int = 2,
) -> vLLM:
    kill_vllm_workers()
    if os.path.exists(os.path.abspath(model)):
        named_arguments.setdefault("served_model_name", model)
        model = os.path.abspath(model)
    port = named_arguments.get("port") or 8000
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((named_arguments.get("host") or "0.0.0.0", port))
            break
        except OSError:
            if "port" in named_arguments and named_arguments["port"] == port:
                raise RuntimeError(f"Port {port} is already in use")
            port += 1
        finally:
            sock.close()
    named_arguments["port"] = port
    args = [
        "vllm",
        "serve",
        model,
        *[
            f"--{key.replace('_', '-')}{f'={value}' if value is not True else ''}"
            for key, value in named_arguments.items()
        ],
        "--api-key=default",
    ]
    # os.system("lsof -ti :8000 | xargs kill -9 2>/dev/null || true")
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            **os.environ,
            **(env or {}),
        },
    )
    if verbosity > 0:
        print(f"$ {' '.join(args)}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log = open(log_file, "w")
    logging = verbosity > 1
    max_concurrent_tokens: Optional[int] = None

    async def log_output(stream: asyncio.StreamReader, io: IO[str]) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded_line = line.decode()
            if logging:
                io.write(decoded_line)
                io.flush()
            log.write(decoded_line)
            log.flush()
            nonlocal max_concurrent_tokens
            if not max_concurrent_tokens:
                match = re.search(
                    r"Maximum concurrency for (\d+) tokens per request: ([\d.]+)x",
                    decoded_line,
                )
                if match:
                    max_concurrent_tokens = int(int(match.group(1)) * float(match.group(2)))
        log.close()

    if process.stdout:
        asyncio.create_task(log_output(process.stdout, sys.stdout))
    if process.stderr:
        asyncio.create_task(log_output(process.stderr, sys.stderr))
    client = AsyncOpenAI(
        api_key="default",
        base_url=f"http://{named_arguments.get('host', '0.0.0.0')}:{named_arguments['port']}/v1",
        max_retries=6,
        http_client=DefaultAsyncHttpxClient(
            limits=httpx.Limits(
                max_connections=max_concurrent_requests,
                max_keepalive_connections=max_concurrent_requests,
            ),
            timeout=httpx.Timeout(timeout=1_200, connect=10.0),
        ),
    )
    start = asyncio.get_event_loop().time()
    while True:
        try:
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model=named_arguments.get("served_model_name", model),
                max_tokens=1,
            )
            break
        except Exception:
            if asyncio.get_event_loop().time() - start > timeout:
                process.terminate()
                kill_vllm_workers()
                raise TimeoutError("vLLM server did not start in time")
            continue
    if logging:
        print(f"vLLM server started succesfully. Logs can be found at {log_file}")
        logging = False
    if max_concurrent_tokens is None:
        process.terminate()
        kill_vllm_workers()
        raise RuntimeError("Max concurrent requests for the maximum model length not logged")
    return vLLM(
        client,
        max_concurrent_tokens,
        named_arguments.get("served_model_name", model),
        process,
    )


def kill_vllm_workers() -> None:
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    pids = [
        line.split()[1]
        for line in result.stdout.splitlines()
        if "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=" in line
    ]
    for pid in pids:
        subprocess.run(["kill", "-9", pid], check=True)
