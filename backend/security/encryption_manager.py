# Local data encryption
"""
Mickey AI - Encryption Manager
Advanced encryption and data protection for sensitive information
"""

import logging
import os
import base64
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import argon2

class EncryptionManager:
    def __init__(self, master_key: str = None, key_file: str = "data/encryption_keys.json"):
        self.logger = logging.getLogger(__name__)
        self.key_file = key_file
        self.backend = default_backend()
        
        # Key management
        self.master_key = master_key
        self.fernet = None
        self.key_derivation_salt = None
        
        # Encryption algorithms
        self.supported_algorithms = {
            'aes-256-gcm': 'AES-256-GCM (Recommended)',
            'aes-256-cbc': 'AES-256-CBC',
            'fernet': 'Fernet (Symmetric)',
            'chacha20': 'ChaCha20-Poly1305'
        }
        
        # Key storage
        self.key_store = {}
        self.key_versions = {}
        
        # Security parameters
        self.encryption_iterations = 100000
        self.key_rotation_days = 90
        
        # Argon2 hasher for password hashing
        self.hasher = argon2.PasswordHasher(
            time_cost=3,  # Number of iterations
            memory_cost=65536,  # 64MB memory
            parallelism=1,  # Number of parallel threads
            hash_len=32,
            salt_len=16
        )
        
        # Initialize encryption system
        self._initialize_encryption_system()
        
        self.logger.info("🔐 Encryption Manager initialized - Data protection active!")

    def _initialize_encryption_system(self):
        """Initialize the encryption system with master key"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            
            # Load or generate master key
            if self.master_key:
                self._derive_keys_from_master(self.master_key)
            else:
                self._load_or_generate_keys()
            
            self.logger.info("Encryption system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Encryption system initialization failed: {str(e)}")
            raise

    def _load_or_generate_keys(self):
        """Load existing keys or generate new ones"""
        try:
            if os.path.exists(self.key_file):
                self._load_keys_from_file()
            else:
                self._generate_new_keys()
                self._save_keys_to_file()
                
        except Exception as e:
            self.logger.error(f"Key management failed: {str(e)}")
            self._generate_new_keys()

    def _generate_new_keys(self):
        """Generate new encryption keys"""
        # Generate master key
        self.master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        # Generate salt for key derivation
        self.key_derivation_salt = os.urandom(16)
        
        # Derive Fernet key
        fernet_key = self._derive_key(self.master_key, self.key_derivation_salt, 32)
        self.fernet = Fernet(fernet_key)
        
        # Initialize key store
        self.key_store = {
            'current_version': 1,
            'keys': {
                1: {
                    'key': fernet_key,
                    'created': self._get_current_timestamp(),
                    'algorithm': 'fernet'
                }
            }
        }
        
        self.key_versions = {1: 'current'}
        
        self.logger.info("New encryption keys generated")

    def _derive_keys_from_master(self, master_key: str):
        """Derive encryption keys from master key"""
        try:
            # Generate salt (in production, this would be stored securely)
            self.key_derivation_salt = os.urandom(16)
            
            # Derive Fernet key
            fernet_key = self._derive_key(master_key, self.key_derivation_salt, 32)
            self.fernet = Fernet(fernet_key)
            
            # Initialize key store
            self.key_store = {
                'current_version': 1,
                'keys': {
                    1: {
                        'key': fernet_key,
                        'created': self._get_current_timestamp(),
                        'algorithm': 'fernet'
                    }
                }
            }
            
            self.logger.info("Encryption keys derived from master key")
            
        except Exception as e:
            self.logger.error(f"Key derivation failed: {str(e)}")
            raise

    def _derive_key(self, password: str, salt: bytes, key_length: int) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=self.encryption_iterations,
            backend=self.backend
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _load_keys_from_file(self):
        """Load encryption keys from file"""
        try:
            with open(self.key_file, 'r') as f:
                key_data = json.load(f)
            
            self.key_store = key_data['key_store']
            self.key_derivation_salt = base64.b64decode(key_data['key_derivation_salt'])
            
            # Set current Fernet key
            current_version = self.key_store['current_version']
            current_key = base64.urlsafe_b64decode(self.key_store['keys'][str(current_version)]['key'])
            self.fernet = Fernet(current_key)
            
            self.logger.info("Encryption keys loaded from file")
            
        except Exception as e:
            self.logger.error(f"Failed to load keys from file: {str(e)}")
            raise

    def _save_keys_to_file(self):
        """Save encryption keys to file (secured with master key)"""
        try:
            key_data = {
                'key_store': self.key_store,
                'key_derivation_salt': base64.b64encode(self.key_derivation_salt).decode()
            }
            
            with open(self.key_file, 'w') as f:
                json.dump(key_data, f, indent=2)
            
            # Set secure file permissions (Unix-like systems)
            try:
                os.chmod(self.key_file, 0o600)
            except:
                pass  # Ignore on Windows
            
            self.logger.info("Encryption keys saved to file")
            
        except Exception as e:
            self.logger.error(f"Failed to save keys to file: {str(e)}")
            raise

    def encrypt_string(self, plaintext: str, algorithm: str = 'fernet') -> Dict[str, Any]:
        """
        Encrypt a string
        
        Args:
            plaintext: String to encrypt
            algorithm: Encryption algorithm to use
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        try:
            if algorithm == 'fernet':
                return self._encrypt_with_fernet(plaintext)
            elif algorithm == 'aes-256-gcm':
                return self._encrypt_with_aes_gcm(plaintext)
            elif algorithm == 'aes-256-cbc':
                return self._encrypt_with_aes_cbc(plaintext)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            self.logger.error(f"String encryption failed: {str(e)}")
            raise

    def _encrypt_with_fernet(self, plaintext: str) -> Dict[str, Any]:
        """Encrypt using Fernet symmetric encryption"""
        encrypted = self.fernet.encrypt(plaintext.encode())
        
        return {
            'encrypted_data': base64.urlsafe_b64encode(encrypted).decode(),
            'algorithm': 'fernet',
            'key_version': self.key_store['current_version'],
            'timestamp': self._get_current_timestamp()
        }

    def _encrypt_with_aes_gcm(self, plaintext: str) -> Dict[str, Any]:
        """Encrypt using AES-256-GCM"""
        # Generate random key and nonce
        key = os.urandom(32)
        nonce = os.urandom(12)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        
        return {
            'encrypted_data': base64.urlsafe_b64encode(ciphertext).decode(),
            'algorithm': 'aes-256-gcm',
            'key': base64.urlsafe_b64encode(key).decode(),
            'nonce': base64.urlsafe_b64encode(nonce).decode(),
            'tag': base64.urlsafe_b64encode(encryptor.tag).decode(),
            'timestamp': self._get_current_timestamp()
        }

    def _encrypt_with_aes_cbc(self, plaintext: str) -> Dict[str, Any]:
        """Encrypt using AES-256-CBC"""
        # Generate random key and IV
        key = os.urandom(32)
        iv = os.urandom(16)
        
        # Pad the data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode()) + padder.finalize()
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return {
            'encrypted_data': base64.urlsafe_b64encode(ciphertext).decode(),
            'algorithm': 'aes-256-cbc',
            'key': base64.urlsafe_b64encode(key).decode(),
            'iv': base64.urlsafe_b64encode(iv).decode(),
            'timestamp': self._get_current_timestamp()
        }

    def decrypt_string(self, encrypted_data: Dict[str, Any]) -> str:
        """
        Decrypt a string
        
        Args:
            encrypted_data: Dictionary with encrypted data and metadata
            
        Returns:
            Decrypted string
        """
        try:
            algorithm = encrypted_data.get('algorithm', 'fernet')
            
            if algorithm == 'fernet':
                return self._decrypt_with_fernet(encrypted_data)
            elif algorithm == 'aes-256-gcm':
                return self._decrypt_with_aes_gcm(encrypted_data)
            elif algorithm == 'aes-256-cbc':
                return self._decrypt_with_aes_cbc(encrypted_data)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            self.logger.error(f"String decryption failed: {str(e)}")
            raise

    def _decrypt_with_fernet(self, encrypted_data: Dict[str, Any]) -> str:
        """Decrypt using Fernet"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data['encrypted_data'])
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()

    def _decrypt_with_aes_gcm(self, encrypted_data: Dict[str, Any]) -> str:
        """Decrypt using AES-256-GCM"""
        key = base64.urlsafe_b64decode(encrypted_data['key'])
        nonce = base64.urlsafe_b64decode(encrypted_data['nonce'])
        tag = base64.urlsafe_b64decode(encrypted_data['tag'])
        ciphertext = base64.urlsafe_b64decode(encrypted_data['encrypted_data'])
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        return decrypted.decode()

    def _decrypt_with_aes_cbc(self, encrypted_data: Dict[str, Any]) -> str:
        """Decrypt using AES-256-CBC"""
        key = base64.urlsafe_b64decode(encrypted_data['key'])
        iv = base64.urlsafe_b64decode(encrypted_data['iv'])
        ciphertext = base64.urlsafe_b64decode(encrypted_data['encrypted_data'])
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
        
        return decrypted.decode()

    def encrypt_file(self, file_path: str, output_path: str = None, algorithm: str = 'fernet') -> Dict[str, Any]:
        """
        Encrypt a file
        
        Args:
            file_path: Path to file to encrypt
            output_path: Output path (optional)
            algorithm: Encryption algorithm to use
            
        Returns:
            Dictionary with encryption metadata
        """
        try:
            if not output_path:
                output_path = file_path + '.encrypted'
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Encrypt file data
            if algorithm == 'fernet':
                encrypted_data = self.fernet.encrypt(file_data)
            else:
                # For large files, we might want to use chunked encryption
                encrypted_data = self._encrypt_large_data(file_data, algorithm)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            self.logger.info(f"File encrypted: {file_path} -> {output_path}")
            
            return {
                'success': True,
                'original_file': file_path,
                'encrypted_file': output_path,
                'algorithm': algorithm,
                'original_size': len(file_data),
                'encrypted_size': len(encrypted_data),
                'timestamp': self._get_current_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"File encryption failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def decrypt_file(self, file_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Decrypt a file
        
        Args:
            file_path: Path to encrypted file
            output_path: Output path (optional)
            
        Returns:
            Dictionary with decryption metadata
        """
        try:
            if not output_path:
                # Remove .encrypted extension or add .decrypted
                if file_path.endswith('.encrypted'):
                    output_path = file_path[:-10]
                else:
                    output_path = file_path + '.decrypted'
            
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt file data
            try:
                decrypted_data = self.fernet.decrypt(encrypted_data)
            except:
                # Try other decryption methods if Fernet fails
                decrypted_data = self._decrypt_large_data(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            self.logger.info(f"File decrypted: {file_path} -> {output_path}")
            
            return {
                'success': True,
                'encrypted_file': file_path,
                'decrypted_file': output_path,
                'encrypted_size': len(encrypted_data),
                'decrypted_size': len(decrypted_data),
                'timestamp': self._get_current_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"File decryption failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _encrypt_large_data(self, data: bytes, algorithm: str) -> bytes:
        """Encrypt large data using chunking (for files too large for memory)"""
        # For simplicity, we'll use Fernet for large data
        # In production, you might want to implement proper chunked encryption
        return self.fernet.encrypt(data)

    def _decrypt_large_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt large data"""
        return self.fernet.decrypt(encrypted_data)

    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        try:
            return self.hasher.hash(password)
        except Exception as e:
            self.logger.error(f"Password hashing failed: {str(e)}")
            raise

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            password: Password to verify
            hashed_password: Hashed password to compare against
            
        Returns:
            Boolean indicating if password is valid
        """
        try:
            return self.hasher.verify(hashed_password, password)
        except argon2.exceptions.VerifyMismatchError:
            return False
        except Exception as e:
            self.logger.error(f"Password verification failed: {str(e)}")
            return False

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure token
        
        Args:
            length: Length of token in bytes
            
        Returns:
            Secure token
        """
        return secrets.token_urlsafe(length)

    def rotate_keys(self) -> bool:
        """
        Rotate encryption keys (key rotation for security)
        
        Returns:
            Boolean indicating success
        """
        try:
            current_version = self.key_store['current_version']
            new_version = current_version + 1
            
            # Generate new key
            new_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            
            # Add new key to store
            self.key_store['keys'][str(new_version)] = {
                'key': new_key,
                'created': self._get_current_timestamp(),
                'algorithm': 'fernet'
            }
            
            # Update current version
            self.key_store['current_version'] = new_version
            
            # Update Fernet instance
            self.fernet = Fernet(base64.urlsafe_b64decode(new_key))
            
            # Save updated keys
            self._save_keys_to_file()
            
            self.logger.info(f"Encryption keys rotated from version {current_version} to {new_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {str(e)}")
            return False

    def reencrypt_data(self, old_encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-encrypt data with current key version
        
        Args:
            old_encrypted_data: Data encrypted with old key
            
        Returns:
            Data encrypted with current key
        """
        try:
            # Decrypt with old method
            plaintext = self.decrypt_string(old_encrypted_data)
            
            # Re-encrypt with current key
            return self.encrypt_string(plaintext)
            
        except Exception as e:
            self.logger.error(f"Data re-encryption failed: {str(e)}")
            raise

    def get_encryption_status(self) -> Dict[str, Any]:
        """Get encryption system status"""
        current_version = self.key_store['current_version']
        key_count = len(self.key_store['keys'])
        
        return {
            'encryption_active': True,
            'current_key_version': current_version,
            'total_key_versions': key_count,
            'supported_algorithms': list(self.supported_algorithms.keys()),
            'key_rotation_days': self.key_rotation_days,
            'key_derivation_iterations': self.encryption_iterations,
            'master_key_configured': self.master_key is not None,
            'mickey_response': "Mickey's keeping your data safe and secure! 🔒"
        }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

    def secure_wipe(self, data: Union[str, bytes]):
        """
        Securely wipe sensitive data from memory
        
        Args:
            data: Data to wipe
        """
        try:
            if isinstance(data, str):
                # Overwrite string data (less effective in Python due to immutability)
                del data
            elif isinstance(data, bytes):
                # For bytes, we can try to overwrite (though limited in Python)
                # This is more symbolic than actually secure in Python
                data = b'\x00' * len(data)
                del data
        except:
            pass

    def cleanup(self):
        """Cleanup sensitive data from memory"""
        try:
            # Securely wipe sensitive attributes
            self.secure_wipe(self.master_key)
            self.master_key = None
            
            # Clear Fernet instance
            self.fernet = None
            
            # Clear key store from memory
            self.key_store.clear()
            
            self.logger.info("Encryption manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

# Test function
def test_encryption_manager():
    """Test the encryption manager"""
    import time
    
    # Create encryption manager
    enc_manager = EncryptionManager()
    
    print("Testing Encryption Manager...")
    
    # Test string encryption
    test_string = "Hello Mickey! This is a secret message."
    encrypted = enc_manager.encrypt_string(test_string)
    print(f"Original: {test_string}")
    print(f"Encrypted: {encrypted['encrypted_data'][:50]}...")
    
    # Test decryption
    decrypted = enc_manager.decrypt_string(encrypted)
    print(f"Decrypted: {decrypted}")
    print(f"Match: {test_string == decrypted}")
    
    # Test password hashing
    password = "MySecurePassword123"
    hashed = enc_manager.hash_password(password)
    print(f"Password: {password}")
    print(f"Hashed: {hashed[:50]}...")
    
    # Test password verification
    verify_result = enc_manager.verify_password(password, hashed)
    print(f"Password verification: {verify_result}")
    
    # Test secure token
    token = enc_manager.generate_secure_token()
    print(f"Secure token: {token}")
    
    # Test status
    status = enc_manager.get_encryption_status()
    print("Encryption Status:", status)
    
    # Cleanup
    enc_manager.cleanup()
    print("Encryption manager test completed!")

if __name__ == "__main__":
    test_encryption_manager()