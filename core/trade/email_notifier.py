# core/trade/email_notifier.py

import smtplib  # ✅ 在文件开头导入
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import os

from core.trade.twap_executor import DailyTWAPPlan


class EmailNotifier:
    """
    邮件通知服务
    """

    def __init__(
            self,
            smtp_server: Optional[str] = None,
            smtp_port: Optional[int] = None,
            sender_email: Optional[str] = None,
            sender_password: Optional[str] = None,
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.qq.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = sender_email or os.getenv("EMAIL_SENDER")
        self.sender_password = sender_password or os.getenv("EMAIL_PASSWORD")

        if not self.sender_email or not self.sender_password:
            raise ValueError("请在 .env 文件中设置 EMAIL_SENDER 和 EMAIL_PASSWORD")

    def send_daily_plan(
            self,
            plan: DailyTWAPPlan,
            receiver_email: str,
    ) -> bool:
        """
        发送每日交易计划
        """
        subject = f"[TradeLz] {plan.symbol} 交易计划 - {plan.date.date()}"

        body = self._build_email_body(plan)

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'html'))

            if self.smtp_port == 465:
                # SSL 加密（163 邮箱）
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
            else:
                # TLS 加密（QQ、Gmail）
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)

            print(f"[OK] 邮件已发送到 {receiver_email}")
            return True

        except Exception as e:
            print(f"[ERROR] 邮件发送失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _build_email_body(self, plan: DailyTWAPPlan) -> str:
        """
        构建 HTML 邮件正文
        """
        if abs(plan.total_delta) < 0.01:
            action_summary = "🟢 今日无需交易"
            color = "#10b981"
        elif plan.total_delta > 0:
            action_summary = f"🔵 今日需要加仓 {abs(plan.total_delta):.2%}"
            color = "#3b82f6"
        else:
            action_summary = f"🔴 今日需要减仓 {abs(plan.total_delta):.2%}"
            color = "#ef4444"

        signals_html = ""
        for sig in plan.signals:
            signals_html += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">{sig.time_window}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; font-weight: bold; color: {color};">{sig.action}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">{sig.absolute_ratio:.2%}</td>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #666; font-size: 13px;">{sig.reason}</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .summary {{ background: #f9fafb; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
                th {{ background: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2 style="margin: 0;">🎯 TradeLz 交易计划</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">{plan.date.date()} | {plan.symbol}</p>
                </div>

                <div class="summary">
                    <h3 style="margin: 0 0 10px 0; color: {color};">{action_summary}</h3>
                    <p style="margin: 5px 0; color: #666;">当前仓位：{plan.current_position:.2%}</p>
                    <p style="margin: 5px 0; color: #666;">目标仓位：{plan.target_position:.2%}</p>
                </div>

                <table>
                    <tr>
                        <th>时间窗口</th>
                        <th>操作</th>
                        <th>交易量</th>
                        <th>说明</th>
                    </tr>
                    {signals_html}
                </table>

                <p style="margin-top: 20px; color: #999; font-size: 13px;">
                    ⚠️ 本邮件仅供参考，请根据实际市场情况调整。
                </p>
            </div>
        </body>
        </html>
        """

        return html
