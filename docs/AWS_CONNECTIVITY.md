# AWS Connectivity

This project connects to AWS only after the orchestrator has routed a user
message to an agent.

- `car_control` sends door and trunk commands to AWS AppSync with
  `mutateVehicle`.
- `car_manual` answers local RAG/database questions. For selected executive
  demo answers, the orchestrator sends an LCD intent command to AWS AppSync
  with `sendVehicleCommand`.

Both paths use the same AppSync endpoint, API key, and vehicle id.

## Required Environment

Set these values in the root `.env` file:

```env
GRAPHQL_HOST=<AppSync GraphQL endpoint>
GRAPHQL_API_KEY=<SOVDGraphQLApi API key>
GRAPHQL_VEHICLE_ID=03174526-5baa-11f0-947d-00155d08e663
```

`docker-compose.yml` passes these variables into `orchestrator` and
`car_control`.

## Car Control Flow

```text
voice/text command
-> orchestrator routes to car_control
-> car_control maps the action to a VSS signal
-> AppSync mutateVehicle
-> AWS publishes to iot/sdv/vehicle_control
```

GraphQL mutation:

```graphql
mutation MutateVehicle($id: ID!, $name: String!, $value: String!) {
  mutateVehicle(id: $id, name: $name, value: $value) {
    id
    name
    value
    success
  }
}
```

Current signal mappings:

| Command family | Action | VSS signal | Value |
|---|---|---|---|
| Driver door | open | `Cabin.Door.Row1.DriverSide.IsOpen` | `true` |
| Driver door | close | `Cabin.Door.Row1.DriverSide.IsOpen` | `false` |
| Driver door | lock | `Cabin.Door.Row1.DriverSide.IsLocked` | `true` |
| Driver door | unlock | `Cabin.Door.Row1.DriverSide.IsLocked` | `false` |
| Rear/driver-side door | open | `Cabin.Door.Row2.DriverSide.IsOpen` | `true` |
| Rear/driver-side door | close | `Cabin.Door.Row2.DriverSide.IsOpen` | `false` |
| Rear/driver-side door | lock | `Cabin.Door.Row2.DriverSide.IsLocked` | `true` |
| Rear/driver-side door | unlock | `Cabin.Door.Row2.DriverSide.IsLocked` | `false` |
| Trunk | open | `Body.Trunk.Rear.IsOpen` | `true` |
| Trunk | close | `Body.Trunk.Rear.IsOpen` | `false` |

## Car Manual FaceCard Flow

```text
voice/text question
-> orchestrator routes to car_manual
-> car_manual returns the matched answer
-> orchestrator maps the answer/question to an LCD intentCode
-> AppSync sendVehicleCommand
-> VehicleCommandTable
-> CommandStreamProcessor
-> VSSDataTable
-> iot/sdv/vehicle_control
-> Goldbox/CAN/LCD
```

GraphQL mutation:

```graphql
mutation SendVehicleCommand($input: VehicleCommandInput!) {
  sendVehicleCommand(input: $input) {
    requestId
    vehicleId
    status
    reasonCode
    success
    timestamp
  }
}
```

Request body:

```json
{
  "input": {
    "requestId": "cmd-lcd-<uuid>",
    "vehicleId": "03174526-5baa-11f0-947d-00155d08e663",
    "commandType": "LCD_INTENT",
    "intentCode": 5,
    "schemaVersion": "1.0"
  }
}
```

Do not send `FaceCard`, `Body.FaceCard`, or `TFT_Answer` directly. AWS
projects `TFT_Request`, `TFT_Answer`, and `FaceCard` into VSS data from the
`intentCode`.

Intent codes:

| Intent code | LCD answer |
|---:|---|
| 1 | Tony |
| 2 | Cherry |
| 3 | Nguyen Duc Kinh / CEO FPT Auto |
| 4 | Pham Minh Tuan / VP FPT |
| 5 | George Leondi / CFO Nissan |
| 6 | Shibata Hideki / CIO Mitsubishi |
| 7 | President Kato / President Mitsubishi |

## Running And Restarting

Check services:

```bash
docker compose -p orch_v1 ps
```

Restart the orchestrator after mounted Python changes under `./orchestrator`:

```bash
docker compose -p orch_v1 restart orchestrator
```

Follow logs:

```bash
docker compose -p orch_v1 logs -f orchestrator
```

No rebuild is needed for mounted Python changes under `./orchestrator`. Rebuild
only when Docker image contents, dependencies, or non-mounted files change.

## Testing Car Control

1. In the AWS IoT MQTT test client, subscribe to:

   ```text
   iot/sdv/vehicle_control
   ```

2. Trigger a supported door or trunk command through the orchestrator:

   ```bash
   curl -sS -X POST http://localhost:8000/v1/orchestrator/message \
     -H 'Content-Type: application/json' \
     -d '{"message":"open the trunk","session_id":"aws-car-control-test"}'
   ```

3. Confirm orchestrator logs route to `car_control` and the car-control agent
   logs a GraphQL response.

4. Confirm AWS IoT receives a message on `iot/sdv/vehicle_control`.

If your running server exposes a different route, check `/health` first and use
the route registered by that server.

## Testing Car Manual FaceCard

1. In the AWS IoT MQTT test client, subscribe to:

   ```text
   iot/sdv/vehicle_control
   ```

2. Trigger a known FaceCard question:

   ```bash
   curl -sS -X POST http://localhost:8000/v1/orchestrator/message \
     -H 'Content-Type: application/json' \
     -d '{"message":"Who is CFO of Nissan?","session_id":"aws-facecard-test"}'
   ```

3. Confirm orchestrator logs show:

   ```text
   Sending face card GraphQL command ... command_type=LCD_INTENT intent_code=5
   Face card GraphQL command result: success=True
   ```

4. Confirm the MQTT payload contains projected fields such as:

   ```text
   Body.FaceCard
   Body.TFT_Request
   Body.TFT_RequestId
   Body.TFT_Answer
   ```

## Troubleshooting

- Missing `GRAPHQL_HOST` or `GRAPHQL_API_KEY`: the FaceCard upload is skipped.
- `sendVehicleCommand success=false`: check `status` and `reasonCode` in the
  GraphQL response.
- GraphQL success but no MQTT message: check the AWS stream processor, VSS
  projection, and IoT publish logs.
- Door/trunk works but FaceCard does not: compare the `sendVehicleCommand`
  request and `intentCode` mapping with the mobile app.
- Old scripts or tests that use `mutateSDV`, `FaceCard`, `Body.FaceCard`, or
  `TFT_Answer` directly are not the mobile-app-compatible FaceCard path.

## Security Notes

- Do not commit API keys.
- Keep `.env` local.
- Use the `SOVDGraphQLApi` API key provided by the AWS/mobile app team.
- Verify the AWS account and region match the MQTT test client.
