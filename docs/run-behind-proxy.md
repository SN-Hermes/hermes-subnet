## Run Behind a Router

If your validator is running behind a home router or NAT, follow the steps below.

1. Update the validator bind address
   
   Open `neurons/validator.py` and navigate to **line 145**. Update the `uvicorn.Config` as follows:
   
   ```python
   config = uvicorn.Config(
       app,
       host="0.0.0.0",
       port="<your internal port>",
       loop="asyncio",
       reload=False,
       log_config=None,   # Disable uvicorn's default logging config
       access_log=False,  # Disable access logs to reduce noise
   )
   
   ```
* Set `host` to `"0.0.0.0"` so the server listens on all local interfaces.
- Set `port` to your **internal (LAN) port**.



2. Configure external IP and port
   
   In `.env.validator`, make sure `EXTERNAL_IP` and `PORT` are set to your **public-facing** IP and port:
   
   ```ini
   EXTERNAL_IP=<your public IP>
   PORT=<your public port>
   ```
- `EXTERNAL_IP`: Your public IP address (WAN IP).

- `PORT`: The public port forwarded by your router to the internal port.

Ensure your router is configured with **port forwarding**, mapping:

<public port>  →  <internal port>


