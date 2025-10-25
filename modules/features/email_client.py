# SMTP email handler
"""
Mickey AI - Email Client
Send, receive, and manage emails with Mickey's personality
"""

import logging
import smtplib
import imaplib
import email
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import os
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

class EmailClient:
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Email configuration
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.imap_server = self.config.get('imap_server', 'imap.gmail.com')
        self.imap_port = self.config.get('imap_port', 993)
        
        # Authentication (should be set via configure method)
        self.email_address = None
        self.password = None
        self.is_configured = False
        
        # Connection objects
        self.smtp_connection = None
        self.imap_connection = None
        
        # Email tracking
        self.sent_emails = []
        self.received_emails = []
        
        # Mickey's email personalities
        self.email_responses = {
            'send_success': [
                "Email sent! Mickey's message is on its way! 📧",
                "Message delivered! Another job well done! ✨",
                "Hot dog! Email sent successfully! 🌭",
                "Mickey's mail magic worked! Message sent! 🐭"
            ],
            'send_failure': [
                "Oops! Mickey couldn't send the email! 😅",
                "Uh oh! The email got lost in the digital maze! 🌀",
                "Mickey's having trouble with the email magic!",
                "The email owls are on strike! Try again? 🦉"
            ],
            'receive_success': [
                "Mickey found your emails! Time to read! 📬",
                "New messages incoming! Mickey's got your mail!",
                "Email treasure hunt successful! Found {} new messages! 🎯",
                "Mickey's mailbox is full of surprises! {} new emails! 🎁"
            ],
            'no_emails': [
                "No new emails! Mickey's mailbox is empty! 📭",
                "All quiet on the email front! No new messages!",
                "Mickey checked - no new emails waiting!",
                "Your inbox is clean! Mickey found zero new messages! ✨"
            ]
        }
        
        self.logger.info("📧 Email Client initialized - Ready to handle messages!")

    def configure(self, email_address: str, password: str, 
                 smtp_server: str = None, imap_server: str = None) -> bool:
        """
        Configure email client with credentials
        
        Args:
            email_address: Email address to use
            password: App password or email password
            smtp_server: SMTP server (optional)
            imap_server: IMAP server (optional)
            
        Returns:
            Boolean indicating success
        """
        try:
            self.email_address = email_address
            self.password = password
            
            if smtp_server:
                self.smtp_server = smtp_server
            if imap_server:
                self.imap_server = imap_server
            
            # Test configuration by connecting
            smtp_success = self._connect_smtp()
            imap_success = self._connect_imap()
            
            self.is_configured = smtp_success or imap_success
            
            if self.is_configured:
                self.logger.info(f"Email client configured for: {email_address}")
            else:
                self.logger.warning("Email configuration failed - limited functionality")
            
            return self.is_configured
            
        except Exception as e:
            self.logger.error(f"Email configuration failed: {str(e)}")
            return False

    def _connect_smtp(self) -> bool:
        """Connect to SMTP server"""
        try:
            self.smtp_connection = smtplib.SMTP(self.smtp_server, self.smtp_port)
            self.smtp_connection.starttls()
            self.smtp_connection.login(self.email_address, self.password)
            self.logger.info("SMTP connection established")
            return True
        except Exception as e:
            self.logger.error(f"SMTP connection failed: {str(e)}")
            self.smtp_connection = None
            return False

    def _connect_imap(self) -> bool:
        """Connect to IMAP server"""
        try:
            self.imap_connection = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            self.imap_connection.login(self.email_address, self.password)
            self.imap_connection.select('inbox')
            self.logger.info("IMAP connection established")
            return True
        except Exception as e:
            self.logger.error(f"IMAP connection failed: {str(e)}")
            self.imap_connection = None
            return False

    def send_email(self, to_address: str, subject: str, body: str, 
                  is_html: bool = False, attachments: List[str] = None) -> Dict[str, Any]:
        """
        Send an email
        
        Args:
            to_address: Recipient email address
            subject: Email subject
            body: Email body content
            is_html: Whether body is HTML
            attachments: List of file paths to attach
            
        Returns:
            Dictionary with send result
        """
        try:
            if not self.is_configured or not self.smtp_connection:
                return self._create_error_response("Email client not configured")

            # Create message
            msg = MimeMultipart()
            msg['From'] = self.email_address
            msg['To'] = to_address
            msg['Subject'] = subject

            # Add body
            if is_html:
                msg.attach(MimeText(body, 'html'))
            else:
                msg.attach(MimeText(body, 'plain'))

            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        self._add_attachment(msg, file_path)
                    else:
                        self.logger.warning(f"Attachment not found: {file_path}")

            # Send email
            self.smtp_connection.send_message(msg)
            
            # Track sent email
            sent_email = {
                'to': to_address,
                'subject': subject,
                'timestamp': datetime.now().isoformat(),
                'attachments': len(attachments) if attachments else 0
            }
            self.sent_emails.append(sent_email)
            
            self.logger.info(f"Email sent to: {to_address}, Subject: {subject}")
            
            return {
                'success': True,
                'action': 'send_email',
                'to': to_address,
                'subject': subject,
                'attachments_count': len(attachments) if attachments else 0,
                'message': f"Email sent to {to_address}",
                'mickey_response': random.choice(self.email_responses['send_success'])
            }
            
        except Exception as e:
            self.logger.error(f"Send email failed: {str(e)}")
            return self._create_error_response(f"Failed to send email: {str(e)}")

    def _add_attachment(self, msg: MimeMultipart, file_path: str):
        """Add file attachment to email"""
        try:
            with open(file_path, "rb") as attachment:
                part = MimeBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            
            filename = os.path.basename(file_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            
            msg.attach(part)
            self.logger.info(f"Attachment added: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to add attachment {file_path}: {str(e)}")

    def receive_emails(self, limit: int = 10, unread_only: bool = True) -> Dict[str, Any]:
        """
        Receive emails from inbox
        
        Args:
            limit: Maximum number of emails to fetch
            unread_only: Whether to fetch only unread emails
            
        Returns:
            Dictionary with received emails
        """
        try:
            if not self.is_configured or not self.imap_connection:
                return self._create_error_response("Email client not configured")

            # Search for emails
            if unread_only:
                status, messages = self.imap_connection.search(None, 'UNSEEN')
            else:
                status, messages = self.imap_connection.search(None, 'ALL')
            
            if status != 'OK':
                return self._create_error_response("Failed to search emails")

            email_ids = messages[0].split()
            email_ids = email_ids[-limit:]  # Get latest emails
            
            emails = []
            for email_id in email_ids:
                email_data = self._fetch_email(email_id)
                if email_data:
                    emails.append(email_data)

            # Update received emails cache
            self.received_emails = emails
            
            self.logger.info(f"Received {len(emails)} emails")
            
            if emails:
                response = random.choice(self.email_responses['receive_success']).format(len(emails))
            else:
                response = random.choice(self.email_responses['no_emails'])
            
            return {
                'success': True,
                'action': 'receive_emails',
                'emails': emails,
                'total_received': len(emails),
                'unread_only': unread_only,
                'message': f"Found {len(emails)} emails",
                'mickey_response': response
            }
            
        except Exception as e:
            self.logger.error(f"Receive emails failed: {str(e)}")
            return self._create_error_response(f"Failed to receive emails: {str(e)}")

    def _fetch_email(self, email_id: bytes) -> Optional[Dict[str, Any]]:
        """Fetch and parse individual email"""
        try:
            status, msg_data = self.imap_connection.fetch(email_id, '(RFC822)')
            
            if status != 'OK':
                return None
            
            raw_email = msg_data[0][1]
            email_message = email.message_from_bytes(raw_email)
            
            # Extract email details
            subject = email_message['subject'] or 'No Subject'
            from_address = email_message['from'] or 'Unknown Sender'
            date = email_message['date'] or datetime.now().isoformat()
            
            # Extract body
            body = self._extract_email_body(email_message)
            
            # Check for attachments
            attachments = self._extract_attachments(email_message)
            
            email_data = {
                'id': email_id.decode(),
                'from': from_address,
                'subject': subject,
                'date': date,
                'body': body[:500] + '...' if len(body) > 500 else body,  # Limit body length
                'attachments_count': len(attachments),
                'attachments': attachments,
                'is_unread': True  # Assuming we're fetching unread
            }
            
            return email_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch email {email_id}: {str(e)}")
            return None

    def _extract_email_body(self, email_message) -> str:
        """Extract text body from email message"""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        body = part.get_payload(decode=True).decode()
                        break
                    except:
                        continue
        else:
            try:
                body = email_message.get_payload(decode=True).decode()
            except:
                body = str(email_message.get_payload())
        
        return body

    def _extract_attachments(self, email_message) -> List[Dict[str, str]]:
        """Extract attachment information from email"""
        attachments = []
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_disposition = str(part.get("Content-Disposition"))
                
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        attachments.append({
                            'filename': filename,
                            'content_type': part.get_content_type(),
                            'size': len(part.get_payload(decode=True)) if part.get_payload(decode=True) else 0
                        })
        
        return attachments

    def mark_as_read(self, email_id: str) -> Dict[str, Any]:
        """Mark email as read"""
        try:
            if not self.imap_connection:
                return self._create_error_response("IMAP connection not available")
            
            status, response = self.imap_connection.store(email_id.encode(), '+FLAGS', '\\Seen')
            
            if status == 'OK':
                return {
                    'success': True,
                    'action': 'mark_as_read',
                    'email_id': email_id,
                    'message': "Email marked as read",
                    'mickey_response': "Mickey marked that email as read! 👀"
                }
            else:
                return self._create_error_response("Failed to mark email as read")
                
        except Exception as e:
            self.logger.error(f"Mark as read failed: {str(e)}")
            return self._create_error_response(f"Failed to mark email as read: {str(e)}")

    def delete_email(self, email_id: str) -> Dict[str, Any]:
        """Delete email"""
        try:
            if not self.imap_connection:
                return self._create_error_response("IMAP connection not available")
            
            status, response = self.imap_connection.store(email_id.encode(), '+FLAGS', '\\Deleted')
            
            if status == 'OK':
                self.imap_connection.expunge()
                return {
                    'success': True,
                    'action': 'delete_email',
                    'email_id': email_id,
                    'message': "Email deleted",
                    'mickey_response': "Mickey tossed that email in the trash! 🗑️"
                }
            else:
                return self._create_error_response("Failed to delete email")
                
        except Exception as e:
            self.logger.error(f"Delete email failed: {str(e)}")
            return self._create_error_response(f"Failed to delete email: {str(e)}")

    def quick_send(self, to_address: str, quick_message: str) -> Dict[str, Any]:
        """
        Send a quick email with Mickey's signature
        
        Args:
            to_address: Recipient email
            quick_message: The message content
            
        Returns:
            Dictionary with send result
        """
        subject = "Message from Mickey AI"
        
        # Add Mickey's signature
        signature = f"""
        
---
Sent by Mickey AI 🐭
Your friendly AI assistant
{datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        
        full_body = quick_message + signature
        
        return self.send_email(to_address, subject, full_body)

    def get_email_stats(self) -> Dict[str, Any]:
        """Get email statistics"""
        return {
            'success': True,
            'email_address': self.email_address,
            'is_configured': self.is_configured,
            'sent_count': len(self.sent_emails),
            'received_count': len(self.received_emails),
            'last_sent': self.sent_emails[-1] if self.sent_emails else None,
            'last_received': self.received_emails[-1] if self.received_emails else None
        }

    def validate_email_address(self, email_address: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email_address) is not None

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'mickey_response': random.choice(self.email_responses['send_failure'])
        }

    def disconnect(self):
        """Disconnect from email servers"""
        try:
            if self.smtp_connection:
                self.smtp_connection.quit()
                self.smtp_connection = None
            
            if self.imap_connection:
                self.imap_connection.logout()
                self.imap_connection = None
            
            self.logger.info("Email connections closed")
            
        except Exception as e:
            self.logger.error(f"Disconnect failed: {str(e)}")

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.disconnect()

# Test function
def test_email_client():
    """Test the email client (requires actual email credentials)"""
    # Note: This is a demo - actual email credentials required for full testing
    client = EmailClient()
    
    # Test configuration (using dummy credentials for demo)
    configured = client.configure(
        email_address="test@example.com",
        password="password",
        smtp_server="smtp.gmail.com",
        imap_server="imap.gmail.com"
    )
    
    print(f"Configuration: {'Success' if configured else 'Failed'}")
    
    # Test email validation
    valid = client.validate_email_address("test@example.com")
    print(f"Email validation: {valid}")
    
    # Test stats
    stats = client.get_email_stats()
    print("Email Stats:", stats)
    
    # Cleanup
    client.disconnect()

if __name__ == "__main__":
    test_email_client()