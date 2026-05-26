"""
Email Agent for EDIATH
Advanced email operations: send, receive, search, manage emails with multiple providers
"""

import asyncio
import email
import imaplib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import logging
import json
from bs4 import BeautifulSoup

# Additional libraries
try:
    import aioimaplib

    AIOIMAP_AVAILABLE = True
except ImportError:
    AIOIMAP_AVAILABLE = False

try:
    from email_validator import validate_email, EmailNotValidError

    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False


class EmailProvider(Enum):
    """Supported email providers"""

    GMAIL = "gmail"
    OUTLOOK = "outlook"
    YAHOO = "yahoo"
    CUSTOM = "custom"


class EmailFolder(Enum):
    """Common email folders"""

    INBOX = "INBOX"
    SENT = "Sent"
    DRAFTS = "Drafts"
    TRASH = "Trash"
    SPAM = "Spam"
    ARCHIVE = "Archive"


class EmailPriority(Enum):
    """Email priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class EmailConfig:
    """Email configuration"""

    provider: EmailProvider
    email_address: str
    password: str
    imap_server: Optional[str] = None
    smtp_server: Optional[str] = None
    imap_port: int = 993
    smtp_port: int = 587
    use_ssl: bool = True
    use_tls: bool = True


@dataclass
class EmailMessage:
    """Email message structure"""

    uid: str
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str]
    bcc_addresses: List[str]
    subject: str
    body: str
    html_body: Optional[str] = None
    attachments: List[Dict] = field(default_factory=list)
    date: datetime = field(default_factory=datetime.now)
    folder: str = "INBOX"
    is_read: bool = False
    priority: EmailPriority = EmailPriority.NORMAL
    size: int = 0
    labels: List[str] = field(default_factory=list)


@dataclass
class Attachment:
    """Email attachment"""

    filename: str
    content: bytes
    content_type: str
    size: int


class EmailAgent:
    """
    Advanced email agent capable of:
    - Sending emails with attachments and HTML content
    - Receiving and fetching emails
    - Searching emails by criteria
    - Managing email folders
    - Email filtering and labeling
    - Batch operations
    - Email templates
    - Email tracking (open/click tracking)
    - Multiple provider support
    - Email parsing and extraction
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Email Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Email configurations
        self.email_configs: Dict[str, EmailConfig] = {}
        self.imap_connections: Dict[str, Any] = {}
        self.smtp_connections: Dict[str, Any] = {}

        # Email templates
        self.templates: Dict[str, str] = {}
        self.template_dir = Path(self.config.get("template_dir", "email_templates"))
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Attachment storage
        self.attachment_dir = Path(
            self.config.get("attachment_dir", "email_attachments")
        )
        self.attachment_dir.mkdir(parents=True, exist_ok=True)

        # Email cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 300)
        self.email_cache: Dict[str, Tuple[datetime, List[EmailMessage]]] = {}

        # Statistics
        self.stats = {
            "emails_sent": 0,
            "emails_received": 0,
            "emails_processed": 0,
            "attachments_downloaded": 0,
            "errors": 0,
            "by_provider": {},
        }

        # Email history
        self.email_history: List[EmailMessage] = []
        self.max_history = self.config.get("max_history", 1000)

        # Load templates
        self._load_templates()

        self.logger.info("Email Agent initialized")

    def _load_templates(self):
        """Load email templates from directory"""
        if self.template_dir.exists():
            for template_file in self.template_dir.glob("*.json"):
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        template_data = json.load(f)
                        template_name = template_file.stem
                        self.templates[template_name] = template_data
                        self.logger.info(f"Loaded template: {template_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load template {template_file}: {e}")

    def _get_provider_settings(self, provider: EmailProvider) -> Dict[str, Any]:
        """Get default settings for email provider"""
        settings = {
            EmailProvider.GMAIL: {
                "imap_server": "imap.gmail.com",
                "smtp_server": "smtp.gmail.com",
                "imap_port": 993,
                "smtp_port": 587,
            },
            EmailProvider.OUTLOOK: {
                "imap_server": "outlook.office365.com",
                "smtp_server": "smtp.office365.com",
                "imap_port": 993,
                "smtp_port": 587,
            },
            EmailProvider.YAHOO: {
                "imap_server": "imap.mail.yahoo.com",
                "smtp_server": "smtp.mail.yahoo.com",
                "imap_port": 993,
                "smtp_port": 587,
            },
        }

        return settings.get(provider, {})

    async def add_account(
        self, account_name: str, config: EmailConfig
    ) -> Dict[str, Any]:
        """
        Add an email account

        Args:
            account_name: Unique name for this account
            config: Email configuration

        Returns:
            Dictionary with account addition result
        """
        if account_name in self.email_configs:
            return {"success": False, "error": f"Account {account_name} already exists"}

        # Validate email
        if EMAIL_VALIDATOR_AVAILABLE:
            try:
                validate_email(config.email_address)
            except EmailNotValidError as e:
                return {"success": False, "error": f"Invalid email address: {str(e)}"}

        # Set provider defaults if not specified
        if config.provider != EmailProvider.CUSTOM:
            defaults = self._get_provider_settings(config.provider)
            if not config.imap_server:
                config.imap_server = defaults["imap_server"]
            if not config.smtp_server:
                config.smtp_server = defaults["smtp_server"]
            if not config.imap_port:
                config.imap_port = defaults["imap_port"]
            if not config.smtp_port:
                config.smtp_port = defaults["smtp_port"]

        self.email_configs[account_name] = config

        # Update statistics
        provider_name = config.provider.value
        if provider_name not in self.stats["by_provider"]:
            self.stats["by_provider"][provider_name] = 0
        self.stats["by_provider"][provider_name] += 1

        self.logger.info(
            f"Added email account: {account_name} ({config.email_address})"
        )

        return {
            "success": True,
            "account_name": account_name,
            "email_address": config.email_address,
            "provider": config.provider.value,
            "message": f"Account {account_name} added successfully",
        }

    async def connect_imap(self, account_name: str) -> Dict[str, Any]:
        """
        Connect to IMAP server

        Args:
            account_name: Name of the account

        Returns:
            Dictionary with connection result
        """
        if account_name not in self.email_configs:
            return {"success": False, "error": f"Account {account_name} not found"}

        if account_name in self.imap_connections:
            # Check if connection is still alive
            try:
                await self.imap_connections[account_name].noop()
                return {"success": True, "message": "Already connected"}
            except:
                # Connection died, reconnect
                del self.imap_connections[account_name]

        config = self.email_configs[account_name]

        try:
            if AIOIMAP_AVAILABLE:
                # Use async IMAP
                imap = aioimaplib.IMAP4_SSL(
                    host=config.imap_server, port=config.imap_port
                )
                await imap.wait_hello_from_server()
                await imap.login(config.email_address, config.password)

                self.imap_connections[account_name] = imap

                return {
                    "success": True,
                    "account_name": account_name,
                    "message": f"Connected to IMAP server for {account_name}",
                }
            else:
                # Use synchronous IMAP (run in executor)
                loop = asyncio.get_event_loop()

                def connect_sync():
                    imap = imaplib.IMAP4_SSL(config.imap_server, config.imap_port)
                    imap.login(config.email_address, config.password)
                    return imap

                imap = await loop.run_in_executor(None, connect_sync)
                self.imap_connections[account_name] = imap

                return {
                    "success": True,
                    "account_name": account_name,
                    "message": f"Connected to IMAP server for {account_name}",
                }

        except Exception as e:
            self.logger.error(f"IMAP connection error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def connect_smtp(self, account_name: str) -> Dict[str, Any]:
        """
        Connect to SMTP server

        Args:
            account_name: Name of the account

        Returns:
            Dictionary with connection result
        """
        if account_name not in self.email_configs:
            return {"success": False, "error": f"Account {account_name} not found"}

        if account_name in self.smtp_connections:
            return {"success": True, "message": "Already connected"}

        config = self.email_configs[account_name]

        try:
            # Create SMTP connection
            if config.use_ssl:
                smtp = smtplib.SMTP_SSL(config.smtp_server, config.smtp_port)
            else:
                smtp = smtplib.SMTP(config.smtp_server, config.smtp_port)

            if config.use_tls and not config.use_ssl:
                smtp.starttls()

            smtp.login(config.email_address, config.password)

            self.smtp_connections[account_name] = smtp

            return {
                "success": True,
                "account_name": account_name,
                "message": f"Connected to SMTP server for {account_name}",
            }

        except Exception as e:
            self.logger.error(f"SMTP connection error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def send_email(
        self,
        account_name: str,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        attachments: Optional[List[Union[str, Path]]] = None,
        priority: EmailPriority = EmailPriority.NORMAL,
        reply_to: Optional[str] = None,
        template_name: Optional[str] = None,
        template_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Send an email

        Args:
            account_name: Account to send from
            to: Recipient email address(es)
            subject: Email subject
            body: Plain text body
            html_body: HTML body (optional)
            cc: CC recipient(s)
            bcc: BCC recipient(s)
            attachments: List of file paths to attach
            priority: Email priority
            reply_to: Reply-to address
            template_name: Name of template to use
            template_data: Data to fill template

        Returns:
            Dictionary with send result
        """
        if account_name not in self.email_configs:
            return {"success": False, "error": f"Account {account_name} not found"}

        # Connect to SMTP if not already connected
        if account_name not in self.smtp_connections:
            connect_result = await self.connect_smtp(account_name)
            if not connect_result["success"]:
                return connect_result

        # Apply template if specified
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            if template_data:
                subject = template["subject"].format(**template_data)
                body = template["body"].format(**template_data)
                if "html_body" in template:
                    html_body = template["html_body"].format(**template_data)

        # Normalize recipients
        to_list = [to] if isinstance(to, str) else to
        cc_list = [cc] if isinstance(cc, str) else cc if cc else []
        bcc_list = [bcc] if isinstance(bcc, str) else bcc if bcc else []

        # Validate emails
        if EMAIL_VALIDATOR_AVAILABLE:
            for email_addr in to_list + cc_list + bcc_list:
                try:
                    validate_email(email_addr)
                except EmailNotValidError as e:
                    return {
                        "success": False,
                        "error": f"Invalid email: {email_addr} - {str(e)}",
                    }

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.email_configs[account_name].email_address
            msg["To"] = ", ".join(to_list)
            msg["Subject"] = subject
            msg["Date"] = email.utils.formatdate(localtime=True)

            if cc_list:
                msg["Cc"] = ", ".join(cc_list)
            if reply_to:
                msg["Reply-To"] = reply_to

            # Add priority
            priority_map = {
                EmailPriority.LOW: "5 (Lowest)",
                EmailPriority.NORMAL: "3 (Normal)",
                EmailPriority.HIGH: "1 (Highest)",
            }
            msg["X-Priority"] = priority_map[priority]

            # Attach body parts
            if body:
                msg.attach(MIMEText(body, "plain"))
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            # Attach files
            if attachments:
                for attachment_path in attachments:
                    path = Path(attachment_path)
                    if not path.exists():
                        self.logger.warning(f"Attachment not found: {attachment_path}")
                        continue

                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition", f'attachment; filename="{path.name}"'
                        )
                        msg.attach(part)

            # Send email
            smtp = self.smtp_connections[account_name]
            all_recipients = to_list + cc_list + bcc_list

            if AIOIMAP_AVAILABLE:
                # Async send (run in executor)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: smtp.send_message(msg, to_addrs=all_recipients)
                )
            else:
                # Sync send
                smtp.send_message(msg, to_addrs=all_recipients)

            # Update statistics
            self.stats["emails_sent"] += 1

            # Add to history
            email_msg = EmailMessage(
                uid=f"sent_{datetime.now().timestamp()}",
                from_address=self.email_configs[account_name].email_address,
                to_addresses=to_list,
                cc_addresses=cc_list,
                bcc_addresses=bcc_list,
                subject=subject,
                body=body,
                html_body=html_body,
                date=datetime.now(),
                folder="Sent",
                priority=priority,
            )
            self._add_to_history(email_msg)

            return {
                "success": True,
                "message_id": msg["Message-ID"],
                "to": to_list,
                "subject": subject,
                "timestamp": datetime.now().isoformat(),
                "message": "Email sent successfully",
            }

        except Exception as e:
            self.logger.error(f"Send email error: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "to": to_list,
                "subject": subject,
            }

    async def fetch_emails(
        self,
        account_name: str,
        folder: EmailFolder = EmailFolder.INBOX,
        limit: int = 50,
        since_date: Optional[datetime] = None,
        unread_only: bool = False,
        mark_as_read: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch emails from IMAP server

        Args:
            account_name: Account to fetch from
            folder: Email folder to fetch from
            limit: Maximum number of emails to fetch
            since_date: Fetch emails since this date
            unread_only: Fetch only unread emails
            mark_as_read: Mark fetched emails as read

        Returns:
            Dictionary with fetched emails
        """
        # Connect to IMAP if not already connected
        if account_name not in self.imap_connections:
            connect_result = await self.connect_imap(account_name)
            if not connect_result["success"]:
                return connect_result

        imap = self.imap_connections[account_name]
        config = self.email_configs[account_name]

        try:
            # Select folder
            if AIOIMAP_AVAILABLE:
                await imap.select(folder.value)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: imap.select(folder.value))

            # Build search criteria
            search_criteria = []
            if since_date:
                search_criteria.append(f'SINCE {since_date.strftime("%d-%b-%Y")}')
            if unread_only:
                search_criteria.append("UNSEEN")

            search_query = " ".join(search_criteria) if search_criteria else "ALL"

            # Search for emails
            if AIOIMAP_AVAILABLE:
                status, data = await imap.search(search_query)
                email_ids = data[0].split()
            else:
                loop = asyncio.get_event_loop()
                status, data = await loop.run_in_executor(
                    None, lambda: imap.search(None, search_query)
                )
                email_ids = data[0].split()

            # Get latest emails
            email_ids = email_ids[-limit:] if limit else email_ids

            emails = []

            for email_id in reversed(email_ids):
                # Fetch email
                if AIOIMAP_AVAILABLE:
                    status, msg_data = await imap.fetch(email_id, "(RFC822)")
                else:
                    loop = asyncio.get_event_loop()
                    status, msg_data = await loop.run_in_executor(
                        None, lambda: imap.fetch(email_id, "(RFC822)")
                    )

                # Parse email
                email_msg = self._parse_email(
                    msg_data,
                    email_id.decode() if isinstance(email_id, bytes) else str(email_id),
                )

                if email_msg:
                    emails.append(email_msg)

                    # Mark as read if requested
                    if mark_as_read and not email_msg.is_read:
                        if AIOIMAP_AVAILABLE:
                            await imap.store(email_id, "+FLAGS", "\\Seen")
                        else:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None, lambda: imap.store(email_id, "+FLAGS", "\\Seen")
                            )
                        email_msg.is_read = True

            # Update statistics
            self.stats["emails_received"] += len(emails)
            self.stats["emails_processed"] += len(emails)

            # Add to history
            for email in emails:
                self._add_to_history(email)

            return {
                "success": True,
                "account": account_name,
                "folder": folder.value,
                "total_fetched": len(emails),
                "emails": [self._email_to_dict(e) for e in emails],
            }

        except Exception as e:
            self.logger.error(f"Fetch emails error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "account": account_name}

    def _parse_email(self, msg_data, uid: str) -> Optional[EmailMessage]:
        """Parse raw email data"""
        try:
            # Parse email
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    raw_email = response_part[1]
                    break
            else:
                return None

            msg = email.message_from_bytes(raw_email)

            # Extract basic fields
            from_addr = self._decode_header(msg.get("From", ""))
            to_addr = self._decode_header(msg.get("To", ""))
            cc_addr = self._decode_header(msg.get("Cc", ""))
            subject = self._decode_header(msg.get("Subject", "No Subject"))
            date_str = msg.get("Date", "")

            # Parse date
            try:
                date = email.utils.parsedate_to_datetime(date_str)
            except:
                date = datetime.now()

            # Extract body and attachments
            body = ""
            html_body = ""
            attachments = []

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    if (
                        content_type == "text/plain"
                        and "attachment" not in content_disposition
                    ):
                        body += self._decode_payload(part)
                    elif (
                        content_type == "text/html"
                        and "attachment" not in content_disposition
                    ):
                        html_body += self._decode_payload(part)
                    elif "attachment" in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            filename = self._decode_header(filename)
                            content = part.get_payload(decode=True)
                            attachments.append(
                                {
                                    "filename": filename,
                                    "size": len(content),
                                    "content_type": content_type,
                                }
                            )
            else:
                # Single part email
                content_type = msg.get_content_type()
                if content_type == "text/plain":
                    body = self._decode_payload(msg)
                elif content_type == "text/html":
                    html_body = self._decode_payload(msg)

            # Clean HTML body
            if html_body:
                soup = BeautifulSoup(html_body, "html.parser")
                if not body:
                    body = soup.get_text()

            # Check if read
            is_read = "\\Seen" not in msg.get("Status", "")

            # Get priority
            priority = EmailPriority.NORMAL
            x_priority = msg.get("X-Priority", "")
            if "1" in x_priority or "High" in x_priority:
                priority = EmailPriority.HIGH
            elif "5" in x_priority or "Low" in x_priority:
                priority = EmailPriority.LOW

            return EmailMessage(
                uid=uid,
                from_address=from_addr,
                to_addresses=self._parse_addresses(to_addr),
                cc_addresses=self._parse_addresses(cc_addr),
                bcc_addresses=[],
                subject=subject,
                body=body.strip(),
                html_body=html_body,
                attachments=attachments,
                date=date,
                is_read=is_read,
                priority=priority,
                size=len(raw_email),
            )

        except Exception as e:
            self.logger.error(f"Parse email error: {str(e)}")
            return None

    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""

        decoded_parts = []
        for part, encoding in email.header.decode_header(header):
            if isinstance(part, bytes):
                try:
                    decoded_parts.append(
                        part.decode(encoding or "utf-8", errors="ignore")
                    )
                except:
                    decoded_parts.append(part.decode("utf-8", errors="ignore"))
            else:
                decoded_parts.append(part)

        return " ".join(decoded_parts)

    def _decode_payload(self, part) -> str:
        """Decode email payload"""
        payload = part.get_payload(decode=True)
        if payload:
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="ignore")
            except:
                return payload.decode("utf-8", errors="ignore")
        return ""

    def _parse_addresses(self, addr_str: str) -> List[str]:
        """Parse email addresses from string"""
        if not addr_str:
            return []

        addresses = email.utils.getaddresses([addr_str])
        return [addr[1] for addr in addresses if addr[1]]

    async def search_emails(
        self,
        account_name: str,
        query: str,
        folder: EmailFolder = EmailFolder.INBOX,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Search emails by criteria

        Args:
            account_name: Account to search
            query: Search query (IMAP search format)
            folder: Folder to search in
            limit: Maximum results

        Returns:
            Dictionary with search results
        """
        # Connect to IMAP if not already connected
        if account_name not in self.imap_connections:
            connect_result = await self.connect_imap(account_name)
            if not connect_result["success"]:
                return connect_result

        imap = self.imap_connections[account_name]

        try:
            # Select folder
            if AIOIMAP_AVAILABLE:
                await imap.select(folder.value)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: imap.select(folder.value))

            # Search
            if AIOIMAP_AVAILABLE:
                status, data = await imap.search(query)
                email_ids = data[0].split()
            else:
                loop = asyncio.get_event_loop()
                status, data = await loop.run_in_executor(
                    None, lambda: imap.search(None, query)
                )
                email_ids = data[0].split()

            email_ids = email_ids[-limit:] if limit else email_ids

            # Fetch found emails
            emails = []
            for email_id in reversed(email_ids):
                if AIOIMAP_AVAILABLE:
                    status, msg_data = await imap.fetch(email_id, "(RFC822)")
                else:
                    loop = asyncio.get_event_loop()
                    status, msg_data = await loop.run_in_executor(
                        None, lambda: imap.fetch(email_id, "(RFC822)")
                    )

                email_msg = self._parse_email(
                    msg_data,
                    email_id.decode() if isinstance(email_id, bytes) else str(email_id),
                )
                if email_msg:
                    emails.append(email_msg)

            return {
                "success": True,
                "query": query,
                "total_found": len(email_ids),
                "returned": len(emails),
                "emails": [self._email_to_dict(e) for e in emails],
            }

        except Exception as e:
            self.logger.error(f"Search emails error: {str(e)}")
            return {"success": False, "error": str(e), "query": query}

    async def mark_as_read(
        self, account_name: str, email_uid: str, folder: EmailFolder = EmailFolder.INBOX
    ) -> Dict[str, Any]:
        """
        Mark email as read

        Args:
            account_name: Account name
            email_uid: Email UID
            folder: Folder containing the email

        Returns:
            Dictionary with operation result
        """
        if account_name not in self.imap_connections:
            connect_result = await self.connect_imap(account_name)
            if not connect_result["success"]:
                return connect_result

        imap = self.imap_connections[account_name]

        try:
            if AIOIMAP_AVAILABLE:
                await imap.select(folder.value)
                await imap.store(email_uid, "+FLAGS", "\\Seen")
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: imap.select(folder.value))
                await loop.run_in_executor(
                    None, lambda: imap.store(email_uid, "+FLAGS", "\\Seen")
                )

            return {
                "success": True,
                "email_uid": email_uid,
                "message": "Marked as read",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def move_email(
        self,
        account_name: str,
        email_uid: str,
        from_folder: EmailFolder,
        to_folder: EmailFolder,
    ) -> Dict[str, Any]:
        """
        Move email between folders

        Args:
            account_name: Account name
            email_uid: Email UID
            from_folder: Source folder
            to_folder: Destination folder

        Returns:
            Dictionary with operation result
        """
        if account_name not in self.imap_connections:
            connect_result = await self.connect_imap(account_name)
            if not connect_result["success"]:
                return connect_result

        imap = self.imap_connections[account_name]

        try:
            if AIOIMAP_AVAILABLE:
                await imap.select(from_folder.value)
                await imap.copy(email_uid, to_folder.value)
                await imap.store(email_uid, "+FLAGS", "\\Deleted")
                await imap.expunge()
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: imap.select(from_folder.value))
                await loop.run_in_executor(
                    None, lambda: imap.copy(email_uid, to_folder.value)
                )
                await loop.run_in_executor(
                    None, lambda: imap.store(email_uid, "+FLAGS", "\\Deleted")
                )
                await loop.run_in_executor(None, lambda: imap.expunge())

            return {
                "success": True,
                "email_uid": email_uid,
                "from_folder": from_folder.value,
                "to_folder": to_folder.value,
                "message": f"Moved email to {to_folder.value}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_email(
        self, account_name: str, email_uid: str, folder: EmailFolder = EmailFolder.INBOX
    ) -> Dict[str, Any]:
        """
        Delete email

        Args:
            account_name: Account name
            email_uid: Email UID
            folder: Folder containing the email

        Returns:
            Dictionary with operation result
        """
        if account_name not in self.imap_connections:
            connect_result = await self.connect_imap(account_name)
            if not connect_result["success"]:
                return connect_result

        imap = self.imap_connections[account_name]

        try:
            if AIOIMAP_AVAILABLE:
                await imap.select(folder.value)
                await imap.store(email_uid, "+FLAGS", "\\Deleted")
                await imap.expunge()
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: imap.select(folder.value))
                await loop.run_in_executor(
                    None, lambda: imap.store(email_uid, "+FLAGS", "\\Deleted")
                )
                await loop.run_in_executor(None, lambda: imap.expunge())

            return {"success": True, "email_uid": email_uid, "message": "Email deleted"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def download_attachment(
        self,
        account_name: str,
        email_uid: str,
        attachment_filename: str,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download email attachment

        Args:
            account_name: Account name
            email_uid: Email UID
            attachment_filename: Name of attachment to download
            save_path: Path to save attachment

        Returns:
            Dictionary with download result
        """
        # Fetch the email first
        result = await self.fetch_emails(account_name, limit=1000)
        if not result["success"]:
            return result

        # Find the email and attachment
        for email_data in result.get("emails", []):
            if email_data.get("uid") == email_uid:
                for attachment in email_data.get("attachments", []):
                    if attachment["filename"] == attachment_filename:
                        # Download attachment logic here
                        # This would require re-fetching the email with attachments
                        pass

        return {"success": False, "error": "Attachment not found"}

    async def create_template(
        self, name: str, subject: str, body: str, html_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create email template

        Args:
            name: Template name
            subject: Email subject template
            body: Email body template
            html_body: HTML body template

        Returns:
            Dictionary with template creation result
        """
        template = {"name": name, "subject": subject, "body": body}

        if html_body:
            template["html_body"] = html_body

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

    def _email_to_dict(self, email: EmailMessage) -> Dict[str, Any]:
        """Convert EmailMessage to dictionary"""
        return {
            "uid": email.uid,
            "from": email.from_address,
            "to": email.to_addresses,
            "cc": email.cc_addresses,
            "subject": email.subject,
            "body": email.body[:500],  # Truncate for display
            "date": email.date.isoformat(),
            "is_read": email.is_read,
            "priority": email.priority.value,
            "attachments": email.attachments,
            "size": email.size,
        }

    def _add_to_history(self, email: EmailMessage):
        """Add email to history"""
        self.email_history.append(email)
        if len(self.email_history) > self.max_history:
            self.email_history = self.email_history[-self.max_history :]

    def get_history(self, limit: int = None) -> List[Dict]:
        """Get email history"""
        history = self.email_history
        if limit:
            history = history[-limit:]

        return [
            {
                "from": e.from_address,
                "subject": e.subject,
                "date": e.date.isoformat(),
                "is_read": e.is_read,
                "folder": e.folder,
            }
            for e in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "history_size": len(self.email_history),
            "active_accounts": len(self.email_configs),
            "active_imap_connections": len(self.imap_connections),
            "active_smtp_connections": len(self.smtp_connections),
            "templates": len(self.templates),
        }

    async def close_connections(self):
        """Close all email connections"""
        for account_name, imap in self.imap_connections.items():
            try:
                if AIOIMAP_AVAILABLE:
                    await imap.logout()
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: imap.logout())
            except:
                pass

        for account_name, smtp in self.smtp_connections.items():
            try:
                smtp.quit()
            except:
                pass

        self.imap_connections.clear()
        self.smtp_connections.clear()

        self.logger.info("All email connections closed")


# Integration wrapper for EDIATH
class EmailAgentWrapper:
    """
    Wrapper class to integrate EmailAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.email_agent = EmailAgent(config)
        self.agent_type = "email"
        self.capabilities = [
            "send_email",
            "fetch_emails",
            "search_emails",
            "manage_folders",
            "email_templates",
            "attachment_handling",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an email request

        Request format:
        {
            'operation': 'send|fetch|search|move|delete|template|history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "add_account":
            provider = EmailProvider(request.get("provider"))
            config = EmailConfig(
                provider=provider,
                email_address=request.get("email"),
                password=request.get("password"),
                imap_server=request.get("imap_server"),
                smtp_server=request.get("smtp_server"),
                imap_port=request.get("imap_port"),
                smtp_port=request.get("smtp_port"),
                use_ssl=request.get("use_ssl", True),
                use_tls=request.get("use_tls", True),
            )

            return await self.email_agent.add_account(
                account_name=request.get("account_name"), config=config
            )

        elif operation == "send":
            priority = request.get("priority", "normal")
            return await self.email_agent.send_email(
                account_name=request.get("account_name"),
                to=request.get("to"),
                subject=request.get("subject"),
                body=request.get("body"),
                html_body=request.get("html_body"),
                cc=request.get("cc"),
                bcc=request.get("bcc"),
                attachments=request.get("attachments"),
                priority=EmailPriority(priority),
                reply_to=request.get("reply_to"),
                template_name=request.get("template_name"),
                template_data=request.get("template_data"),
            )

        elif operation == "fetch":
            folder = request.get("folder", "INBOX")
            since_date = request.get("since_date")
            if since_date:
                since_date = datetime.fromisoformat(since_date)

            return await self.email_agent.fetch_emails(
                account_name=request.get("account_name"),
                folder=EmailFolder(folder),
                limit=request.get("limit", 50),
                since_date=since_date,
                unread_only=request.get("unread_only", False),
                mark_as_read=request.get("mark_as_read", False),
            )

        elif operation == "search":
            return await self.email_agent.search_emails(
                account_name=request.get("account_name"),
                query=request.get("query"),
                folder=EmailFolder(request.get("folder", "INBOX")),
                limit=request.get("limit", 50),
            )

        elif operation == "mark_read":
            return await self.email_agent.mark_as_read(
                account_name=request.get("account_name"),
                email_uid=request.get("email_uid"),
                folder=EmailFolder(request.get("folder", "INBOX")),
            )

        elif operation == "move":
            return await self.email_agent.move_email(
                account_name=request.get("account_name"),
                email_uid=request.get("email_uid"),
                from_folder=EmailFolder(request.get("from_folder")),
                to_folder=EmailFolder(request.get("to_folder")),
            )

        elif operation == "delete":
            return await self.email_agent.delete_email(
                account_name=request.get("account_name"),
                email_uid=request.get("email_uid"),
                folder=EmailFolder(request.get("folder", "INBOX")),
            )

        elif operation == "template":
            action = request.get("action", "create")
            if action == "create":
                return await self.email_agent.create_template(
                    name=request.get("name"),
                    subject=request.get("subject"),
                    body=request.get("body"),
                    html_body=request.get("html_body"),
                )
            else:
                return {"success": False, "error": f"Unknown template action: {action}"}

        elif operation == "history":
            return {
                "success": True,
                "history": self.email_agent.get_history(limit=request.get("limit")),
            }

        elif operation == "stats":
            return self.email_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "EmailAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.email_agent.get_stats(),
            "supported_providers": [p.value for p in EmailProvider],
        }

    async def close(self):
        """Clean up resources"""
        await self.email_agent.close_connections()


# Example usage and testing
async def test_email_agent():
    """Test the email agent functionality"""

    # Initialize agent
    agent = EmailAgent()

    print("=== Email Agent Test ===\n")

    # Test adding an account (using environment variables or config)
    print("1. Adding Email Account...")
    print("   (Note: Use valid credentials for actual testing)")

    # This is a placeholder - use actual credentials in real testing
    # config = EmailConfig(
    #     provider=EmailProvider.GMAIL,
    #     email_address="your_email@gmail.com",
    #     password="your_password",
    # )
    # result = await agent.add_account("test_account", config)
    # print(f"   Account added: {result['success']}")

    # Test template creation
    print("\n2. Creating Email Template...")
    result = await agent.create_template(
        name="welcome",
        subject="Welcome {name}!",
        body="Hello {name},\n\nWelcome to our service!\n\nBest regards,\nThe Team",
        html_body="<h1>Welcome {name}!</h1><p>Welcome to our service!</p>",
    )
    print(f"   Template created: {result['success']}")

    # Test sending email (commented for safety)
    print("\n3. Sending Test Email...")
    print("   (Email sending would happen here with valid credentials)")

    # Get statistics
    print("\n4. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Templates: {stats['templates']}")
    print(f"   Emails sent: {stats['emails_sent']}")
    print(f"   History size: {stats['history_size']}")

    # Close connections
    await agent.close_connections()

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_email_agent())
