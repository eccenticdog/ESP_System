---
name: offline-service-navigation
description: Find nearby offline service stations and generate navigation guidance for service visits or destination trips. Use when the task is about service-center lookup, repair-shop discovery, geolocation parsing, or map navigation.
---

# offline-service-navigation

Use this skill when the user needs:

- Nearby repair or service station lookup
- Brand-specific service center filtering
- Current-location resolution from natural language
- Offline destination navigation

## Inputs

The bundled script supports two modes:

- `service-station`: resolve the user's location, query nearby repair shops, optionally filter by brand, and try to generate a navigation link for the closest match
- `poi-nav`: resolve the user's location and generate navigation info for a normal destination

## Invocation

Run the bundled script from the repo root or from the `backend/app` directory.

```powershell
python its_multi_agent/backend/app/skills/offline-service-navigation-1.0.0/scripts/invoke_service_navigation.py service-station --query "我在昌平区，帮我找最近的联想服务站" --brand 联想
```

```powershell
python its_multi_agent/backend/app/skills/offline-service-navigation-1.0.0/scripts/invoke_service_navigation.py poi-nav --query "我在昌平区" --destination "清华大学"
```

## Notes

- This skill reuses the project's existing local service-station and MCP map capabilities.
- Required runtime dependencies come from the application itself, including database config and Baidu Map MCP access.
- If the environment is missing DB credentials, map AK, or network access, the script returns a structured error payload instead of crashing.
