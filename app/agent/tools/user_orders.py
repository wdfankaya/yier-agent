from app.agent.tools.decorator import tool
from app.agent.tools.mock_data import ORDERS

STATUS_LABELS = {
    "pending": "待发货",
    "shipped": "已发货",
    "delivered": "已签收",
    "refund_processing": "退款中",
}


@tool(
    desc=(
        "查询当前用户的所有订单概要列表（订单号、状态、商品、金额、下单时间）。"
        "当用户想查订单但未提供订单号，或提供的订单号查不到时，"
        "调用此工具列出订单供用户确认"
    ),
)
def list_user_orders() -> dict:
    """查询当前用户的所有订单概要列表。"""
    orders = [
        {
            "order_id": o["order_id"],
            "status": STATUS_LABELS.get(o["status"], o["status"]),
            "items_summary": "、".join(item["name"] for item in o["items"]),
            "total": o["total"],
            "created_at": o["created_at"],
        }
        for o in ORDERS.values()
    ]
    return {"success": True, "count": len(orders), "orders": orders}
