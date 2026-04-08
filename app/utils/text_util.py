TOOL_NAME_MAPPING = {
    "bailian_web_search": "联网搜索",
    "search_mcp": "联网搜索",
    "map_geocode": "地址解析",
    "map_ip_location": "IP定位",
    "map_search_places": "地点搜索",
    "map_uri": "生成导航链接",
    "baidu_map_mcp": "百度地图查询",
    "query_knowledge": "查询知识库",
    "resolve_user_location_from_text": "位置解析",
    "query_nearest_repair_shops_by_coords": "查询附近服务站",
    "geocode_address": "地址转坐标",
    "offline_service_navigation_skill": "线下服务与导航 Skill",
    "consult_technical_expert": "咨询技术专家",
    "query_service_station_and_navigate": "服务站与地理位置专家",
}


def format_tool_call_html(tool_name: str) -> str:
    display_name = TOOL_NAME_MAPPING.get(tool_name, tool_name)
    return f"""
<div class="tech-process-card tool-call">
    <div class="tech-process-header">
        <span class="tech-icon">📧</span>
        <span class="tech-label">正在调用工具</span>
    </div>
    <div class="tech-process-flow">
        <span class="tech-node source">调度中心</span>
        <span class="tech-arrow">→</span>
        <span class="tech-node target">{display_name}</span>
    </div>
</div>
"""


def format_agent_update_html(agent_name: str) -> str:
    return f"""
<div class="tech-process-card agent-update">
    <div class="tech-process-header">
        <span class="tech-icon">🤖</span>
        <span class="tech-label">智能体切换</span>
    </div>
    <div class="tech-process-body">
        <span class="tech-text">当前接管: <strong class="highlight">{agent_name}</strong></span>
    </div>
</div>
"""
