jetson-containers run \
  -e HUGGINGFACE_TOKEN=hf_xxx \
  -e FORCE_BUILD=on \
  -v $(pwd)/jetson_test/llama_gptq_free.sh:/opt/llama_gptq_free.sh \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /opt/llama_gptq_free.sh

# Benchmark TTFT & Tokens/sec:
jetson-containers run \
  -v $(pwd)/jetson_test/benchmark_ttft_tps.sh:/opt/benchmark_ttft_tps.sh \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /opt/benchmark_ttft_tps.sh

  cd /home/acevn/ai-demo-edge-main/llama-cpp/models

wget "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf" \
  -O qwen3-0.6b-q4_k_m.gguf

hf download   bartowski/Llama-3.2-1B-Instruct-GGUF   --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf"   --local-dir .

source venv/bin/activate

cd /home/acevn/ai-demo-edge-main/jetson-containers

./jetson-containers run \
  -v /mnt/nvme/Projects/orchestrator-on-edge/llama-cpp/models:/models \
  $(./autotag llama_cpp)

llama-server \
  -m /models/qwen3_sft_merged.Q4_K_M.gguf \
  --jinja \
  --port 8080 \
  --host 0.0.0.0 \
  -ngl 99 \
  -t 12 \
  -fa \
  --log-colors \
  --parallel 1 \
  --n-predict 4096 \
  --ctx-size 8192 \
  --batch-size 8192 \
  --cpu-range 0-11 \
  --cpu-strict 1

  docker compose up orchestrator

   docker compose -p orch_v1 up car_manual

   docker compose -p orch_v1 build --no-cache

84:D3:52:CB:E7:25

navigation-1      |     stream = await self._network_backend.connect_tcp(**kwargs)
navigation-1      |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpcore/_backends/auto.py", line 31, in connect_tcp
navigation-1      |     return await self._backend.connect_tcp(
navigation-1      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
navigation-1      |     ...<5 lines>...
navigation-1      |     )
navigation-1      |     ^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpcore/_backends/anyio.py", line 113, in connect_tcp
navigation-1      |     with map_exceptions(exc_map):
navigation-1      |          ~~~~~~~~~~~~~~^^^^^^^^^
navigation-1      |   File "/root/.local/share/uv/python/cpython-3.14.3-linux-aarch64-gnu/lib/python3.14/contextlib.py", line 162, in __exit__
navigation-1      |     self.gen.throw(value)
navigation-1      |     ~~~~~~~~~~~~~~^^^^^^^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
navigation-1      |     raise to_exc(exc) from exc
navigation-1      | httpcore.ConnectError: All connection attempts failed
navigation-1      | 
navigation-1      | The above exception was the direct cause of the following exception:
navigation-1      | 
navigation-1      | Traceback (most recent call last):
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/openai/_base_client.py", line 1604, in request
navigation-1      |     response = await self._client.send(
navigation-1      |                ^^^^^^^^^^^^^^^^^^^^^^^^
navigation-1      |     ...<3 lines>...
navigation-1      |     )
navigation-1      |     ^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpx/_client.py", line 1629, in send
navigation-1      |     response = await self._send_handling_auth(
navigation-1      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
navigation-1      |     ...<4 lines>...
navigation-1      |     )
navigation-1      |     ^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpx/_client.py", line 1657, in _send_handling_auth
navigation-1      |     response = await self._send_handling_redirects(
navigation-1      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
navigation-1      |     ...<3 lines>...
navigation-1      |     )
navigation-1      |     ^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpx/_client.py", line 1694, in _send_handling_redirects
navigation-1      |     response = await self._send_single_request(request)
navigation-1      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpx/_client.py", line 1730, in _send_single_request
navigation-1      |     response = await transport.handle_async_request(request)
navigation-1      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpx/_transports/default.py", line 393, in handle_async_request
navigation-1      |     with map_httpcore_exceptions():
navigation-1      |          ~~~~~~~~~~~~~~~~~~~~~~~^^
navigation-1      |   File "/root/.local/share/uv/python/cpython-3.14.3-linux-aarch64-gnu/lib/python3.14/contextlib.py", line 162, in __exit__
navigation-1      |     self.gen.throw(value)
navigation-1      |     ~~~~~~~~~~~~~~^^^^^^^
navigation-1      |   File "/app/.venv/lib/python3.14/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
navigation-1      |     raise mapped_exc(message) from exc
navigation-1      | httpx.ConnectError: All connection attempts failed
navigation-1      | 07:36:49 | DEBUG    | openai._base_client | Raising connection error
navigation-1      | 07:36:49 | ERROR    | navigation.llm | LLM intent parse failed: Connection error.

   
  
