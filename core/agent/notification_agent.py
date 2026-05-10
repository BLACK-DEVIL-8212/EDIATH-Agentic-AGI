"""
Notification Agent for EDIATH
Multi-channel notification system: desktop, webhook, email, SMS, Slack, Discord, Telegram, and more
"""

import asyncio
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import hmac
import hashlib

# Desktop notifications
try:
    from plyer import notification

    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

try:
    import websocket

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Email for notifications (reuse or separate)
try:
    import smtplib
    from email.mime.text import MIMEText

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = True  # Built-in


class NotificationChannel(Enum):
    """Supported notification channels"""

    DESKTOP = "desktop"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    TWILIO = "twilio"
    PUSHOVER = "pushover"
    WEB_SOCKET = "websocket"
    CUSTOM = "custom"


class NotificationPriority(Enum):
    """Notification priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """Notification delivery status"""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"


@dataclass
class Notification:
    """Notification structure"""

    id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority
    timestamp: datetime
    status: NotificationStatus
    data: Dict[str, Any] = field(default_factory=dict)
    recipient: Optional[str] = None
    retry_count: int = 0
    delivered_at: Optional[datetime] = None


@dataclass
class ChannelConfig:
    """Channel configuration"""

    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class NotificationAgent:
    """
    Advanced notification agent capable of:
    - Multi-channel notifications (desktop, webhook, email, SMS, etc.)
    - Priority-based routing
    - Retry logic with exponential backoff
    - Notification templates
    - Batch notifications
    - Scheduled notifications
    - Delivery tracking
    - Channel health monitoring
    - Rate limiting per channel
    - Webhook integrations
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Notification Agent

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Channel configurations
        self.channels: Dict[NotificationChannel, ChannelConfig] = {}
        self._init_channels()

        # Rate limiting
        self.rate_limits: Dict[NotificationChannel, List[datetime]] = {}
        self.rate_limit_config = self.config.get(
            "rate_limits",
            {
                "desktop": 10,  # notifications per minute
                "webhook": 60,
                "email": 30,
                "slack": 30,
                "discord": 30,
                "telegram": 20,
            },
        )

        # Retry configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 5)  # seconds
        self.retry_backoff = self.config.get("retry_backoff", 2)  # multiplier

        # Notification queue
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self.queue_worker = None

        # Notification history
        self.notification_history: List[Notification] = []
        self.max_history = self.config.get("max_history", 10000)

        # Templates
        self.templates: Dict[str, Dict] = {}
        self.template_dir = Path(
            self.config.get("template_dir", "notification_templates")
        )
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_notifications": 0,
            "successful_notifications": 0,
            "failed_notifications": 0,
            "by_channel": {},
            "by_priority": {},
            "average_delivery_time": 0.0,
        }

        # Webhook secrets for verification
        self.webhook_secrets: Dict[str, str] = self.config.get("webhook_secrets", {})

        # Load templates
        self._load_templates()

        # Start queue processor
        self.start_queue_processor()

        self.logger.info("Notification Agent initialized")

    def _init_channels(self):
        """Initialize notification channels"""
        # Desktop notifications
        self.channels[NotificationChannel.DESKTOP] = ChannelConfig(
            channel=NotificationChannel.DESKTOP,
            enabled=PLYER_AVAILABLE,
            config=self.config.get("desktop_config", {}),
        )

        # Webhook
        self.channels[NotificationChannel.WEBHOOK] = ChannelConfig(
            channel=NotificationChannel.WEBHOOK,
            enabled=True,
            config=self.config.get("webhook_config", {}),
        )

        # Email
        self.channels[NotificationChannel.EMAIL] = ChannelConfig(
            channel=NotificationChannel.EMAIL,
            enabled=True,
            config=self.config.get("email_config", {}),
        )

        # Slack
        self.channels[NotificationChannel.SLACK] = ChannelConfig(
            channel=NotificationChannel.SLACK,
            enabled=bool(self.config.get("slack_webhook_url")),
            config=self.config.get("slack_config", {}),
        )

        # Discord
        self.channels[NotificationChannel.DISCORD] = ChannelConfig(
            channel=NotificationChannel.DISCORD,
            enabled=bool(self.config.get("discord_webhook_url")),
            config=self.config.get("discord_config", {}),
        )

        # Telegram
        self.channels[NotificationChannel.TELEGRAM] = ChannelConfig(
            channel=NotificationChannel.TELEGRAM,
            enabled=bool(self.config.get("telegram_bot_token")),
            config=self.config.get("telegram_config", {}),
        )

        # Twilio (SMS)
        self.channels[NotificationChannel.TWILIO] = ChannelConfig(
            channel=NotificationChannel.TWILIO,
            enabled=bool(self.config.get("twilio_account_sid")),
            config=self.config.get("twilio_config", {}),
        )

        # Pushover
        self.channels[NotificationChannel.PUSHOVER] = ChannelConfig(
            channel=NotificationChannel.PUSHOVER,
            enabled=bool(self.config.get("pushover_token")),
            config=self.config.get("pushover_config", {}),
        )

        # WebSocket
        self.channels[NotificationChannel.WEB_SOCKET] = ChannelConfig(
            channel=NotificationChannel.WEB_SOCKET,
            enabled=WEBSOCKET_AVAILABLE,
            config=self.config.get("websocket_config", {}),
        )

    def _load_templates(self):
        """Load notification templates"""
        if self.template_dir.exists():
            for template_file in self.template_dir.glob("*.json"):
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        template_data = json.load(f)
                        template_name = template_file.stem
                        self.templates[template_name] = template_data
                        self.logger.info(
                            f"Loaded notification template: {template_name}"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to load template {template_file}: {e}")

    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is rate limited"""
        if channel not in self.rate_limits:
            self.rate_limits[channel] = []

        limit = self.rate_limit_config.get(channel.value, 60)
        now = datetime.now()

        # Clean old entries
        self.rate_limits[channel] = [
            ts for ts in self.rate_limits[channel] if (now - ts).seconds < 60
        ]

        # Check limit
        if len(self.rate_limits[channel]) >= limit:
            return False

        self.rate_limits[channel].append(now)
        return True

    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"notif_{timestamp}"

    async def send_notification(
        self,
        title: str,
        message: str,
        channel: Union[NotificationChannel, List[NotificationChannel]],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        recipient: Optional[str] = None,
        data: Optional[Dict] = None,
        template_name: Optional[str] = None,
        template_data: Optional[Dict] = None,
        wait_for_delivery: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a notification

        Args:
            title: Notification title
            message: Notification message
            channel: Single channel or list of channels
            priority: Notification priority
            recipient: Recipient identifier (email, phone, etc.)
            data: Additional data for the notification
            template_name: Name of template to use
            template_data: Data to fill template
            wait_for_delivery: Wait for delivery confirmation

        Returns:
            Dictionary with notification result
        """
        # Apply template if specified
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            if template_data:
                title = (
                    template["title"].format(**template_data)
                    if "title" in template
                    else title
                )
                message = template["message"].format(**template_data)

        # Handle multiple channels
        if isinstance(channel, list):
            results = []
            for ch in channel:
                result = await self.send_notification(
                    title,
                    message,
                    ch,
                    priority,
                    recipient,
                    data,
                    template_name,
                    template_data,
                    wait_for_delivery,
                )
                results.append(result)

            return {
                "success": any(r["success"] for r in results),
                "notifications": results,
                "total": len(results),
                "successful": sum(1 for r in results if r["success"]),
            }

        # Create notification object
        notification_id = self._generate_notification_id()
        notification = Notification(
            id=notification_id,
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            timestamp=datetime.now(),
            status=NotificationStatus.PENDING,
            data=data or {},
            recipient=recipient,
        )

        # Check if channel is enabled
        channel_config = self.channels.get(channel)
        if not channel_config or not channel_config.enabled:
            notification.status = NotificationStatus.FAILED
            self._add_to_history(notification)
            return {
                "success": False,
                "notification_id": notification_id,
                "error": f"Channel {channel.value} is not enabled",
                "channel": channel.value,
            }

        # Check rate limit
        if not self._check_rate_limit(channel):
            notification.status = NotificationStatus.FAILED
            self._add_to_history(notification)
            return {
                "success": False,
                "notification_id": notification_id,
                "error": "Rate limit exceeded",
                "channel": channel.value,
            }

        # Send notification based on channel
        start_time = datetime.now()

        try:
            if channel == NotificationChannel.DESKTOP:
                result = await self._send_desktop(notification)
            elif channel == NotificationChannel.WEBHOOK:
                result = await self._send_webhook(notification)
            elif channel == NotificationChannel.EMAIL:
                result = await self._send_email(notification)
            elif channel == NotificationChannel.SLACK:
                result = await self._send_slack(notification)
            elif channel == NotificationChannel.DISCORD:
                result = await self._send_discord(notification)
            elif channel == NotificationChannel.TELEGRAM:
                result = await self._send_telegram(notification)
            elif channel == NotificationChannel.TWILIO:
                result = await self._send_twilio(notification)
            elif channel == NotificationChannel.PUSHOVER:
                result = await self._send_pushover(notification)
            elif channel == NotificationChannel.WEB_SOCKET:
                result = await self._send_websocket(notification)
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported channel: {channel.value}",
                }

            delivery_time = (datetime.now() - start_time).total_seconds()

            if result["success"]:
                notification.status = NotificationStatus.DELIVERED
                notification.delivered_at = datetime.now()
                self.stats["successful_notifications"] += 1
            else:
                notification.status = NotificationStatus.FAILED
                self.stats["failed_notifications"] += 1
                # Attempt retry if needed
                if notification.retry_count < self.max_retries:
                    await self._retry_notification(notification)

            # Update statistics
            self.stats["total_notifications"] += 1

            channel_name = channel.value
            if channel_name not in self.stats["by_channel"]:
                self.stats["by_channel"][channel_name] = 0
            self.stats["by_channel"][channel_name] += 1

            priority_name = priority.value
            if priority_name not in self.stats["by_priority"]:
                self.stats["by_priority"][priority_name] = 0
            self.stats["by_priority"][priority_name] += 1

            # Update average delivery time
            total_time = self.stats["average_delivery_time"] * (
                self.stats["total_notifications"] - 1
            )
            self.stats["average_delivery_time"] = (
                total_time + delivery_time
            ) / self.stats["total_notifications"]

            self._add_to_history(notification)

            return {
                "success": result["success"],
                "notification_id": notification_id,
                "channel": channel.value,
                "priority": priority.value,
                "delivery_time": delivery_time,
                "error": result.get("error"),
                "timestamp": notification.timestamp.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Notification error: {str(e)}")
            notification.status = NotificationStatus.FAILED
            self.stats["failed_notifications"] += 1
            self._add_to_history(notification)

            return {
                "success": False,
                "notification_id": notification_id,
                "error": str(e),
                "channel": channel.value,
            }

    async def _send_desktop(self, notification: Notification) -> Dict[str, Any]:
        """Send desktop notification"""
        if not PLYER_AVAILABLE:
            return {
                "success": False,
                "error": "Plyer not available for desktop notifications",
            }

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: notification.notify(
                    title=notification.title,
                    message=notification.message,
                    app_name="EDIATH",
                    timeout=10,
                ),
            )

            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_webhook(self, notification: Notification) -> Dict[str, Any]:
        """Send webhook notification"""
        webhook_url = notification.data.get("webhook_url") or self.config.get(
            "default_webhook_url"
        )

        if not webhook_url:
            return {"success": False, "error": "Webhook URL not configured"}

        try:
            # Prepare payload
            payload = {
                "id": notification.id,
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "timestamp": notification.timestamp.isoformat(),
                "data": notification.data,
            }

            # Add signature if secret configured
            if notification.id in self.webhook_secrets:
                secret = self.webhook_secrets[notification.id]
                signature = hmac.new(
                    secret.encode(), json.dumps(payload).encode(), hashlib.sha256
                ).hexdigest()
                payload["signature"] = signature

            # Send webhook
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                ),
            )

            if response.status_code in [200, 201, 202, 204]:
                return {"success": True}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_email(self, notification: Notification) -> Dict[str, Any]:
        """Send email notification"""
        email_config = self.channels[NotificationChannel.EMAIL].config

        smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
        smtp_port = email_config.get("smtp_port", 587)
        sender_email = email_config.get("sender_email")
        sender_password = email_config.get("sender_password")

        if not sender_email or not sender_password:
            return {"success": False, "error": "Email credentials not configured"}

        recipient = notification.recipient or email_config.get("default_recipient")
        if not recipient:
            return {"success": False, "error": "No recipient specified"}

        try:
            # Create email
            msg = MIMEText(notification.message)
            msg["Subject"] = notification.title
            msg["From"] = sender_email
            msg["To"] = recipient

            # Send email
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._send_email_sync(
                    smtp_server,
                    smtp_port,
                    sender_email,
                    sender_password,
                    recipient,
                    msg,
                ),
            )

            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _send_email_sync(
        self, smtp_server, smtp_port, sender_email, sender_password, recipient, msg
    ):
        """Synchronous email sending"""
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

    async def _send_slack(self, notification: Notification) -> Dict[str, Any]:
        """Send Slack notification"""
        webhook_url = self.config.get("slack_webhook_url")

        if not webhook_url:
            return {"success": False, "error": "Slack webhook URL not configured"}

        # Determine color based on priority
        colors = {
            NotificationPriority.LOW: "#808080",
            NotificationPriority.NORMAL: "#3498db",
            NotificationPriority.HIGH: "#f39c12",
            NotificationPriority.URGENT: "#e74c3c",
        }

        # Prepare Slack payload
        payload = {
            "attachments": [
                {
                    "color": colors[notification.priority],
                    "title": notification.title,
                    "text": notification.message,
                    "footer": "EDIATH Notification System",
                    "ts": int(notification.timestamp.timestamp()),
                }
            ]
        }

        # Add fields if data provided
        if notification.data:
            fields = []
            for key, value in notification.data.items():
                fields.append({"title": key, "value": str(value), "short": True})
            payload["attachments"][0]["fields"] = fields

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.post(webhook_url, json=payload, timeout=10)
            )

            if response.status_code == 200:
                return {"success": True}
            else:
                return {"success": False, "error": f"Slack error: {response.text}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_discord(self, notification: Notification) -> Dict[str, Any]:
        """Send Discord notification"""
        webhook_url = self.config.get("discord_webhook_url")

        if not webhook_url:
            return {"success": False, "error": "Discord webhook URL not configured"}

        # Determine color based on priority
        colors = {
            NotificationPriority.LOW: 8421504,
            NotificationPriority.NORMAL: 3447003,
            NotificationPriority.HIGH: 15844367,
            NotificationPriority.URGENT: 15548997,
        }

        # Prepare Discord payload
        payload = {
            "embeds": [
                {
                    "title": notification.title,
                    "description": notification.message,
                    "color": colors[notification.priority],
                    "timestamp": notification.timestamp.isoformat(),
                    "footer": {"text": "EDIATH Notification System"},
                }
            ]
        }

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.post(webhook_url, json=payload, timeout=10)
            )

            if response.status_code in [200, 204]:
                return {"success": True}
            else:
                return {"success": False, "error": f"Discord error: {response.text}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_telegram(self, notification: Notification) -> Dict[str, Any]:
        """Send Telegram notification"""
        bot_token = self.config.get("telegram_bot_token")
        chat_id = notification.recipient or self.config.get("telegram_chat_id")

        if not bot_token or not chat_id:
            return {"success": False, "error": "Telegram configuration missing"}

        # Prepare message
        message = f"*{notification.title}*\n\n{notification.message}"

        # Add priority indicator
        priority_emoji = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.NORMAL: "📌",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.URGENT: "🚨",
        }
        message = f"{priority_emoji[notification.priority]} {message}"

        # Send via Telegram Bot API
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.post(url, json=payload, timeout=10)
            )

            if response.status_code == 200:
                return {"success": True}
            else:
                return {"success": False, "error": f"Telegram error: {response.text}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_twilio(self, notification: Notification) -> Dict[str, Any]:
        """Send SMS via Twilio"""
        account_sid = self.config.get("twilio_account_sid")
        auth_token = self.config.get("twilio_auth_token")
        from_number = self.config.get("twilio_from_number")

        if not account_sid or not auth_token or not from_number:
            return {"success": False, "error": "Twilio configuration missing"}

        recipient = notification.recipient
        if not recipient:
            return {"success": False, "error": "No recipient phone number specified"}

        try:
            from twilio.rest import Client

            loop = asyncio.get_event_loop()
            client = Client(account_sid, auth_token)

            message = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    body=f"{notification.title}\n{notification.message}",
                    from_=from_number,
                    to=recipient,
                ),
            )

            return {"success": message.status in ["queued", "sent", "delivered"]}

        except ImportError:
            return {"success": False, "error": "Twilio library not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_pushover(self, notification: Notification) -> Dict[str, Any]:
        """Send Pushover notification"""
        token = self.config.get("pushover_token")
        user_key = notification.recipient or self.config.get("pushover_user_key")

        if not token or not user_key:
            return {"success": False, "error": "Pushover configuration missing"}

        # Priority mapping
        priority_map = {
            NotificationPriority.LOW: -1,
            NotificationPriority.NORMAL: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.URGENT: 2,
        }

        payload = {
            "token": token,
            "user": user_key,
            "title": notification.title,
            "message": notification.message,
            "priority": priority_map[notification.priority],
        }

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    "https://api.pushover.net/1/messages.json", data=payload, timeout=10
                ),
            )

            if response.status_code == 200:
                return {"success": True}
            else:
                return {"success": False, "error": f"Pushover error: {response.text}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_websocket(self, notification: Notification) -> Dict[str, Any]:
        """Send WebSocket notification"""
        if not WEBSOCKET_AVAILABLE:
            return {"success": False, "error": "WebSocket library not available"}

        ws_url = notification.data.get("websocket_url") or self.config.get(
            "websocket_url"
        )

        if not ws_url:
            return {"success": False, "error": "WebSocket URL not configured"}

        try:
            payload = {
                "id": notification.id,
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "timestamp": notification.timestamp.isoformat(),
                "data": notification.data,
            }

            loop = asyncio.get_event_loop()

            def send_ws():
                ws = websocket.WebSocket()
                ws.connect(ws_url)
                ws.send(json.dumps(payload))
                ws.close()

            await loop.run_in_executor(None, send_ws)

            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _retry_notification(self, notification: Notification):
        """Retry failed notification with exponential backoff"""
        notification.retry_count += 1
        delay = self.retry_delay * (
            self.retry_backoff ** (notification.retry_count - 1)
        )

        self.logger.info(
            f"Retrying notification {notification.id} (attempt {notification.retry_count})"
        )

        await asyncio.sleep(delay)

        # Resend
        result = await self.send_notification(
            notification.title,
            notification.message,
            notification.channel,
            notification.priority,
            notification.recipient,
            notification.data,
        )

        if not result["success"] and notification.retry_count < self.max_retries:
            # Schedule another retry
            asyncio.create_task(self._retry_notification(notification))

    async def send_batch(self, notifications: List[Dict]) -> Dict[str, Any]:
        """
        Send multiple notifications in batch

        Args:
            notifications: List of notification dictionaries

        Returns:
            Dictionary with batch results
        """
        tasks = []
        for notif in notifications:
            task = self.send_notification(
                title=notif["title"],
                message=notif["message"],
                channel=notif["channel"],
                priority=notif.get("priority", NotificationPriority.NORMAL),
                recipient=notif.get("recipient"),
                data=notif.get("data"),
                template_name=notif.get("template_name"),
                template_data=notif.get("template_data"),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))

        return {
            "success": successful > 0,
            "total": len(notifications),
            "successful": successful,
            "failed": len(notifications) - successful,
            "results": results,
        }

    async def schedule_notification(
        self,
        delay_seconds: int,
        title: str,
        message: str,
        channel: NotificationChannel,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Schedule a notification for later delivery

        Args:
            delay_seconds: Delay in seconds
            title: Notification title
            message: Notification message
            channel: Notification channel
            **kwargs: Additional parameters for send_notification

        Returns:
            Dictionary with scheduling result
        """

        async def delayed_send():
            await asyncio.sleep(delay_seconds)
            return await self.send_notification(title, message, channel, **kwargs)

        # Schedule the task
        task = asyncio.create_task(delayed_send())

        return {
            "success": True,
            "scheduled": True,
            "delay_seconds": delay_seconds,
            "scheduled_time": (datetime.now().timestamp() + delay_seconds),
            "message": f"Notification scheduled in {delay_seconds} seconds",
        }

    async def create_template(
        self, name: str, title: str, message: str
    ) -> Dict[str, Any]:
        """
        Create notification template

        Args:
            name: Template name
            title: Title template (can include {variables})
            message: Message template (can include {variables})

        Returns:
            Dictionary with template creation result
        """
        template = {
            "name": name,
            "title": title,
            "message": message,
            "created_at": datetime.now().isoformat(),
        }

        # Save template
        template_file = self.template_dir / f"{name}.json"
        try:
            with open(template_file, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2)

            self.templates[name] = template

            return {
                "success": True,
                "name": name,
                "message": f"Template {name} created",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_queue_processor(self):
        """Start the notification queue processor"""
        if not self.processing:
            self.processing = True
            self.queue_worker = asyncio.create_task(self._process_queue())
            self.logger.info("Notification queue processor started")

    async def stop_queue_processor(self):
        """Stop the notification queue processor"""
        self.processing = False
        if self.queue_worker:
            self.queue_worker.cancel()
            try:
                await self.queue_worker
            except asyncio.CancelledError:
                pass
            self.logger.info("Notification queue processor stopped")

    async def _process_queue(self):
        """Process queued notifications"""
        while self.processing:
            try:
                # Get notification from queue
                notification_data = await self.notification_queue.get()

                # Send notification
                result = await self.send_notification(**notification_data)

                # Log result
                if not result["success"]:
                    self.logger.warning(
                        f"Queued notification failed: {result.get('error')}"
                    )

                self.notification_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Queue processor error: {str(e)}")
                await asyncio.sleep(1)

    async def queue_notification(self, **notification_data) -> Dict[str, Any]:
        """
        Queue a notification for async processing

        Args:
            **notification_data: Same parameters as send_notification

        Returns:
            Dictionary with queuing result
        """
        await self.notification_queue.put(notification_data)

        return {
            "success": True,
            "queued": True,
            "queue_size": self.notification_queue.qsize(),
            "message": "Notification queued for processing",
        }

    def _add_to_history(self, notification: Notification):
        """Add notification to history"""
        self.notification_history.append(notification)
        if len(self.notification_history) > self.max_history:
            self.notification_history = self.notification_history[-self.max_history :]

    def get_history(
        self,
        limit: int = None,
        channel: Optional[NotificationChannel] = None,
        status: Optional[NotificationStatus] = None,
    ) -> List[Dict]:
        """Get notification history"""
        history = self.notification_history

        if channel:
            history = [n for n in history if n.channel == channel]

        if status:
            history = [n for n in history if n.status == status]

        if limit:
            history = history[-limit:]

        return [
            {
                "id": n.id,
                "title": n.title,
                "message": n.message[:100],
                "channel": n.channel.value,
                "priority": n.priority.value,
                "status": n.status.value,
                "timestamp": n.timestamp.isoformat(),
                "delivered_at": n.delivered_at.isoformat() if n.delivered_at else None,
            }
            for n in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        success_rate = (
            (
                self.stats["successful_notifications"]
                / self.stats["total_notifications"]
                * 100
            )
            if self.stats["total_notifications"] > 0
            else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "history_size": len(self.notification_history),
            "queue_size": self.notification_queue.qsize(),
            "active_channels": [
                c.value for c, cfg in self.channels.items() if cfg.enabled
            ],
            "templates": len(self.templates),
        }

    def clear_history(self):
        """Clear notification history"""
        self.notification_history.clear()
        self.logger.info("Notification history cleared")


# Integration wrapper for EDIATH
class NotificationAgentWrapper:
    """
    Wrapper class to integrate NotificationAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.notification_agent = NotificationAgent(config)
        self.agent_type = "notification"
        self.capabilities = [
            "send_notification",
            "batch_notifications",
            "scheduled_notifications",
            "queue_notifications",
            "template_management",
            "multi_channel_support",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a notification request

        Request format:
        {
            'operation': 'send|batch|schedule|queue|template|history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "send":
            channel = request.get("channel")
            if isinstance(channel, str):
                channel = NotificationChannel(channel)
            elif isinstance(channel, list):
                channel = [NotificationChannel(c) for c in channel]

            priority = request.get("priority", "normal")

            return await self.notification_agent.send_notification(
                title=request.get("title"),
                message=request.get("message"),
                channel=channel,
                priority=NotificationPriority(priority),
                recipient=request.get("recipient"),
                data=request.get("data"),
                template_name=request.get("template_name"),
                template_data=request.get("template_data"),
                wait_for_delivery=request.get("wait_for_delivery", False),
            )

        elif operation == "batch":
            notifications = request.get("notifications", [])
            # Convert channel strings to enum in each notification
            for notif in notifications:
                if "channel" in notif:
                    if isinstance(notif["channel"], str):
                        notif["channel"] = NotificationChannel(notif["channel"])
                if "priority" in notif and isinstance(notif["priority"], str):
                    notif["priority"] = NotificationPriority(notif["priority"])

            return await self.notification_agent.send_batch(notifications)

        elif operation == "schedule":
            channel = NotificationChannel(request.get("channel"))
            priority = request.get("priority", "normal")

            return await self.notification_agent.schedule_notification(
                delay_seconds=request.get("delay_seconds"),
                title=request.get("title"),
                message=request.get("message"),
                channel=channel,
                priority=NotificationPriority(priority),
                recipient=request.get("recipient"),
                data=request.get("data"),
                template_name=request.get("template_name"),
                template_data=request.get("template_data"),
            )

        elif operation == "queue":
            channel = request.get("channel")
            if isinstance(channel, str):
                channel = NotificationChannel(channel)

            priority = request.get("priority", "normal")

            return await self.notification_agent.queue_notification(
                title=request.get("title"),
                message=request.get("message"),
                channel=channel,
                priority=NotificationPriority(priority),
                recipient=request.get("recipient"),
                data=request.get("data"),
                template_name=request.get("template_name"),
                template_data=request.get("template_data"),
            )

        elif operation == "template":
            action = request.get("action", "create")
            if action == "create":
                return await self.notification_agent.create_template(
                    name=request.get("name"),
                    title=request.get("title"),
                    message=request.get("message"),
                )
            else:
                return {"success": False, "error": f"Unknown template action: {action}"}

        elif operation == "history":
            channel = request.get("channel")
            if channel:
                channel = NotificationChannel(channel)

            status = request.get("status")
            if status:
                status = NotificationStatus(status)

            return {
                "success": True,
                "history": self.notification_agent.get_history(
                    limit=request.get("limit"), channel=channel, status=status
                ),
            }

        elif operation == "stats":
            return self.notification_agent.get_stats()

        elif operation == "clear_history":
            self.notification_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "NotificationAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.notification_agent.get_stats(),
            "supported_channels": [c.value for c in NotificationChannel],
        }

    async def close(self):
        """Clean up resources"""
        await self.notification_agent.stop_queue_processor()


# Example usage and testing
async def test_notification_agent():
    """Test the notification agent functionality"""

    # Initialize agent
    agent = NotificationAgent()

    print("=== Notification Agent Test ===\n")

    # Test desktop notification (if available)
    print("1. Testing Desktop Notification...")
    if PLYER_AVAILABLE:
        result = await agent.send_notification(
            title="EDIATH Test",
            message="This is a test notification from EDIATH",
            channel=NotificationChannel.DESKTOP,
            priority=NotificationPriority.NORMAL,
        )
        print(f"   Desktop notification: {result['success']}")
    else:
        print("   Desktop notifications not available (install plyer)")

    # Test template creation
    print("\n2. Creating Notification Template...")
    result = await agent.create_template(
        name="system_alert",
        title="System Alert: {alert_type}",
        message="Alert: {message}\nSeverity: {severity}\nTime: {time}",
    )
    print(f"   Template created: {result['success']}")

    # Test batch notifications
    print("\n3. Testing Batch Notifications...")
    notifications = [
        {
            "title": "Batch Test 1",
            "message": "This is batch notification 1",
            "channel": (
                NotificationChannel.DESKTOP
                if PLYER_AVAILABLE
                else NotificationChannel.WEBHOOK
            ),
            "priority": NotificationPriority.NORMAL,
        },
        {
            "title": "Batch Test 2",
            "message": "This is batch notification 2",
            "channel": (
                NotificationChannel.DESKTOP
                if PLYER_AVAILABLE
                else NotificationChannel.WEBHOOK
            ),
            "priority": NotificationPriority.HIGH,
        },
    ]

    result = await agent.send_batch(notifications)
    print(f"   Batch results: {result['successful']}/{result['total']} successful")

    # Test scheduled notification
    print("\n4. Scheduling Notification...")
    result = await agent.schedule_notification(
        delay_seconds=5,
        title="Scheduled Test",
        message="This notification was scheduled 5 seconds ago",
        channel=(
            NotificationChannel.DESKTOP
            if PLYER_AVAILABLE
            else NotificationChannel.WEBHOOK
        ),
    )
    print(f"   Scheduled: {result['success']}, delay: {result['delay_seconds']}s")

    # Wait for scheduled notification
    await asyncio.sleep(6)

    # Get statistics
    print("\n5. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total notifications: {stats['total_notifications']}")
    print(f"   Successful: {stats['successful_notifications']}")
    print(f"   Failed: {stats['failed_notifications']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Active channels: {', '.join(stats['active_channels'])}")

    # Get history
    print("\n6. Notification History...")
    history = agent.get_history(limit=5)
    for notif in history:
        print(f"   [{notif['timestamp'][:19]}] {notif['title']} - {notif['status']}")

    # Clean up
    await agent.stop_queue_processor()

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_notification_agent())
