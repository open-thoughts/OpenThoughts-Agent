from daytona import Daytona

SANDBOX_ID = "9d9c3bfa-a5d5-45f7-ae5a-54d24efbf8ec"

daytona = Daytona()  # reads DAYTONA_API_KEY / DAYTONA_API_URL env vars
sandbox = daytona.get(SANDBOX_ID)
sandbox.delete(timeout=0)  # timeout=0 = wait until the API finishes
print("Sandbox deleted.")