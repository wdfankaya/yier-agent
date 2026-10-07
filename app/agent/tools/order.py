from app.agent.tools.decorator import tool
from app.agent.tools.mock_data import ORDERS


@tool(
    desc="根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等",
    params={"order_id": "订单号，例如 ORD-20240115-001"},
)
def query_order(order_id: str) -> dict:
    """根据订单号查询订单详情，包括状态、商品、金额、物流等信息。"""
    order = ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"未找到订单 {order_id}，请核实订单号"}
    return {"success": True, "order": order}
