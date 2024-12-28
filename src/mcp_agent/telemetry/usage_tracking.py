from mcp_agent.config import settings


def send_usage_data():
    if settings.disable_usage_telemetry:
        print("Usage tracking disabled")
        return

    # TODO: saqadri - implement usage tracking
    # data = {"installation_id": str(uuid.uuid4()), "version": "0.1.0"}
    # try:
    #     requests.post("https://telemetry.example.com/usage", json=data, timeout=2)
    # except:
    #     pass
